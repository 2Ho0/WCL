import os
import shutil
import wandb  # Weights & Biases 로깅 라이브러리
import gym  # 강화학습 환경 라이브러리
from gym_minigrid.wrappers import *  # MiniGrid 환경 관련 래퍼
import dreamerv2.api as dv2  # DreamerV2 API
from input_args import parse_minigrid_args  # 실행 시 입력 인자 처리

# ✅ TensorFlow GPU 설정 (메모리 동적 할당)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # TensorFlow의 GPU 메모리 제한 해제

# ✅ Weights & Biases 로그인 (API 키 필요)
wandb.login(key="251cbda79884670a14d9e45aa41195e7e6730c92")

# 🔹 MiniGrid 학습 실행 함수
def run_minigrid(args):
    tag = args.tag + "_" + str(args.seed)  # 태그 + 랜덤 시드 값 설정

    # ✅ DreamerV2 기본 설정값을 업데이트하여 학습 설정 구성
    config = dv2.defaults.update({
        'logdir': '{0}/minigrid_{1}'.format(args.logdir, tag),  # 로그 저장 경로
        'log_every': 1e3,  # 1000 스텝마다 로그 기록
        'log_every_video': 2e5,  # 200,000 스텝마다 동영상 저장
        'train_every': 10,  # 10 스텝마다 학습 수행
        'prefill': 1e4,  # 학습 전 10,000 스텝 동안 데이터 수집
        'time_limit': 100,  # 최대 에피소드 길이
        'actor_ent': 3e-3,  # 정책 엔트로피 조정
        'loss_scales.kl': 1.0,  # KL 손실 스케일링
        'discount': 0.99,  # 할인율
        'steps': args.steps,  # 전체 학습 스텝 수
        'cl': args.cl,  # Continual Learning 사용 여부
        'num_tasks': args.num_tasks,  # 학습할 태스크 개수
        'num_task_repeats': args.num_task_repeats,  # 태스크 반복 횟수
        'seed': args.seed,  # 랜덤 시드
        'eval_every': 1e4,  # 10,000 스텝마다 평가 수행
        'eval_steps': 1e3,  # 평가 시 1,000 스텝 사용
        'tag': tag,  # 태그 설정
        "unbalanced_steps": args.unbalanced_steps,
        'replay.capacity': args.replay_capacity,  # 리플레이 버퍼 용량
        'replay.reservoir_sampling': args.reservoir_sampling,  # 리저버 샘플링 사용 여부
        'replay.recent_past_sampl_thres': args.recent_past_sampl_thres,  # 최근 샘플 임계값
        'sep_exp_eval_policies': args.sep_exp_eval_policies,  # 평가 시 개별 정책 사용 여부
        'replay.minlen': args.minlen,  # 리플레이 버퍼 최소 길이
        'wandb.group': args.wandb_group,  # W&B 그룹
        'wandb.name': "mwilliam55",  # W&B 실험 이름
        'wandb.project': "wcl",  # W&B 프로젝트 이름
        'wandb.entity': 'hyeonglee-dku',  # W&B 엔터티 설정
    }).parse_flags()

    # 🔹 Plan2Explore (탐색 중심 학습) 활성화 시 추가 설정
    if args.plan2explore:
        config = config.update({
            'wandb.entity': 'hyeonglee-dku',
            'expl_behavior': 'Plan2Explore',
            'pred_discount': args.rssm_full_recon,
            'grad_heads': ['decoder', 'reward', 'discount'] if args.rssm_full_recon else ['decoder'],
            'expl_intr_scale': args.expl_intr_scale,
            'expl_extr_scale': args.expl_extr_scale,
            'expl_every': args.expl_every,
            'discount': 0.99,
            'wandb.name': "mwilliam55"
        }).parse_flags()

    # ✅ W&B 실험 시작
    wandb.init(
        config=config,
        reinit=True,
        resume=False,
        sync_tensorboard=True,
        dir=args.wandb_dir,
        **config.wandb,
    )

    # 🔹 Continual Learning 모드 (CL 학습 실행)
    if config.cl:
        env_names = [
            'MiniGrid-DoorKey-8x8-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]

        envs = []
        for i in range(config.num_tasks):
            name = env_names[i]
            env = gym.make(name)
            env = RGBImgPartialObsWrapper(env)  # 'mission' 필드 제거
            if args.state_bonus:
                assert not args.plan2explore, "state bonus와 plan2explore는 동시에 사용할 수 없음"
                env = StateBonus(env)
            envs.append(env)

        # 🔹 평가 환경 설정 (MultiSkill Task 포함)
        if args.eval_skills:
            env_names.append('MiniGrid-MultiSkill-N2-v0')  # 다중 기술 평가 환경 추가

        eval_envs = []
        for name in env_names:
            env = gym.make(name)
            env = RGBImgPartialObsWrapper(env)
            eval_envs.append(env)

        dv2.cl_train_loop(envs, config, eval_envs=eval_envs)

    # 🔹 단일 환경 학습 모드
    else:
        env_names = [
            'MiniGrid-DoorKey-8x8-v0',
            'MiniGrid-LavaCrossingS9N1-v0',
            'MiniGrid-SimpleCrossingS9N1-v0',
        ]

        name = env_names[args.env]
        env = gym.make(name)
        env = RGBImgPartialObsWrapper(env)
        dv2.train(env, config)

    # ✅ 학습 데이터 삭제 (옵션)
    if args.del_exp_replay:
        shutil.rmtree(os.path.join(config['logdir'], 'train_episodes'))

    # ✅ W&B 로깅 종료
    wandb.finish()

# 🔹 실행 코드
if __name__ == "__main__":
    args = parse_minigrid_args()  # 실행 시 입력된 인자 파싱
    run_minigrid(args)  # MiniGrid 학습 실행
