import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import gym
import copy
import ast

from dreamerv2.expl import Plan2Explore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import ruamel.yaml as yaml

import agent
import common

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

from ruamel.yaml import YAML

yaml = YAML(typ='safe', pure=True)
configs = yaml.load((pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def train(env, config, outputs=None):

    # set seeds
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    print('GPU available? ', tf.test.is_gpu_available())

    outputs = outputs or [
        common.TerminalOutput(),
        common.JSONLOutput(config.logdir),
        common.TensorBoardOutput(logdir=config.logdir,skipped_metrics=config.skipped_metrics),
    ]
    replay = common.Replay(logdir / 'train_episodes', **config.replay)
    step = common.Counter(replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)
    replay.logger = logger 

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every_video)
    should_expl = common.Until(config.expl_until) # config.expl_until == 0 then we are always exploring

    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        if should_video(step):
            for key in config.log_keys_video:
                logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    env = common.GymWrapper(env)
    env = common.ResizeImage(env)
    if hasattr(env.act_space['action'], 'n'):
        env = common.OneHotAction(env)
    else:
        env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)

    driver = common.Driver([env])
    driver.on_episode(per_episode)
    driver.on_step(lambda tran, worker: step.increment())
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)

    prefill = max(0, config.prefill - replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(env.act_space)
        driver(random_agent, steps=prefill, episodes=1)
        driver.reset()

    print('Create agent.')
    agnt = agent.Agent(config, env.obs_space, env.act_space, step)

    if isinstance(agnt._expl_behavior, Plan2Explore):
        replay.agent = agnt

    dataset = iter(replay.dataset(**config.dataset))
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(dataset))
    if (logdir / 'variables.pkl').exists():
        print("Loading agent.")
        agnt.load(logdir / 'variables.pkl')
    else:
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            train_agent(next(dataset))
    policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            if should_video(step):
                logger.add(agnt.report(next(dataset)))
            logger.write(fps=True)

    def eval_per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        # print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('eval_return', score)
        logger.scalar('eval_length', length)
        if should_video(step):
            for key in config.log_keys_video:
                logger.video(f'eval_{step.value}', ep[key])
        logger.write()

    driver.on_step(train_step)

    eval_driver = common.Driver([env])
    eval_driver.on_episode(eval_per_episode)  # cl eval loop
    # in the original api the evaluation policy and the training policy are the same
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    while step < config.steps:
        logger.write()
        driver(policy, steps=config.eval_every)
        if config.sep_exp_eval_policies:
            eval_driver(eval_policy, steps=config.eval_steps)
        else:
            eval_driver(policy, steps=config.eval_steps)
        agnt.save(logdir / 'variables.pkl')

def cl_train_loop(envs, config, outputs=None, eval_envs=None):
    # 랜덤 시드 설정 (재현성을 위해)
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    # 불균형한 학습 스텝을 설정 (클래스 불균형이 있을 경우 고려)
    unbalanced_steps = ast.literal_eval(config.unbalanced_steps)

    # 로그 디렉토리 설정 및 생성
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    print('GPU available? ', tf.test.is_gpu_available())

    # 로그 출력 형식 지정
    outputs = outputs or [
        common.TerminalOutput(),
        common.JSONLOutput(config.logdir),
        common.TensorBoardOutput(logdir=config.logdir, skipped_metrics=config.skipped_metrics),
    ]

    # 리플레이 버퍼 생성 (환경 별로 저장)
    replay = common.Replay(
        logdir / 'train_episodes', **config.replay, num_tasks=config.num_tasks) # train_episodes 폴더에 경험 데이터 저장
    

    total_step = common.Counter(replay.stats['total_steps']) # replay buffer에 저장된 총 스텝 수를 가져와서 현재 학습 스텝을 초기화
    print("Replay buffer total steps: {}".format(replay.stats['total_steps']))
    logger = common.Logger(total_step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)
    replay.logger = logger
    
    # 현재 진행 중인 작업과 진행된 스텝 계산
    if unbalanced_steps is not None: # 태스크 별로 스텝이 다를 경우
        tot_steps_after_task = np.cumsum(unbalanced_steps) # 각 태스크가 끝나는 누적 스텝을 계산
        #현재까지 학습된 총 스텝을 기준으로 어떤 태스크를 학습해야 하는지 결정
        # 현재 학습된 스텝 수가 각 태스크의 최대 스텝보다 작은 경우의 인덱스를 찾음 -> 아직 완료되지 않은 태스크를 현재 태스크로 설정
        task_id = next((i for i, j in enumerate(replay.stats['total_steps'] < tot_steps_after_task) if j), None)
        print("Task {}".format(task_id))
        # 현재 몇 번째 태스크 학습 반복인지 계산
        # 학습된 총 스텝을 전체 태스크의 총 스텝으로 나누어 몇 번째 반복인지 계산
        rep = int(replay.stats['total_steps'] // (np.sum(unbalanced_steps)))
        print("Rep {}".format(rep))
        #현재 진행중인 태스크에서 몇 번째 스텝부터 다시 시작해야 하는지 계산
        restart_step = (unbalanced_steps-tot_steps_after_task+replay.stats['total_steps'])[task_id]
    else: # 모든 태스크가 동일한 스텝을 가질 경우
        task_id = int(replay.stats['total_steps'] // config.steps) # 현재 태스크 결정
        print("Task {}".format(task_id))
        rep = int(replay.stats['total_steps'] // (config.steps * config.num_tasks)) # 몇 번째 반복인지 계산
        print("Rep {}".format(rep))
        restart_step = int(replay.stats['total_steps'] % config.steps) # 다시 시작할 스텝 계산
    print("Restart step: {}".format(restart_step))
    restart = True if restart_step > 0 else False # restart가 True면 기존 학습 이어서 진행, False면 새로운 태스크 학습을 처음부터 시작작

    # 학습 및 로깅 조건 설정
    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every_video)
    should_video_eval = common.Every(config.log_every_video)
    if config.expl_every:
        print("exploring every {} steps".format(config.expl_every))
        should_expl = common.Every(config.expl_every)
    else:
        should_expl = common.Until(config.expl_until)
    should_recon = common.Every(config.log_recon_every)

    # 에피소드 종료 시 실행되는 함수 정의
    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        logger.scalar('task', task_id)
        logger.scalar('replay_capacity', replay.stats['loaded_steps'])
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        if should_video(total_step):
            for key in config.log_keys_video:
                logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    # 환경을 생성하고, 필요한 드라이버 설정
    def create_envs_drivers(_env):
        _env = common.GymWrapper(_env)
        _env = common.ResizeImage(_env)
        if hasattr(_env.act_space['action'], 'n'):
            _env = common.OneHotAction(_env)
        else:
            _env = common.NormalizeAction(_env)
        _env = common.TimeLimit(_env, config.time_limit)

        driver = common.Driver([_env])
        driver.on_episode(per_episode)
        driver.on_step(lambda tran, worker: total_step.increment())
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)
        driver.on_step(lambda tran, worker: step.increment())
        return _env, driver

    # 평가 환경 설정 (없으면 학습 환경과 동일하게 설정)
    if eval_envs is None:
        eval_envs = envs

    _eval_envs = []
    for i in range(len(eval_envs)):
        env, _ = create_envs_drivers(eval_envs[i])
        _eval_envs.append(env)

    # 연속 학습 루프 실행
    while rep < config.num_task_repeats: # 전체 태스크 학습을 몇 번 반복할지
        while task_id < len(envs): # 각 태스크를 차례대로 학습
            print("\n\t Task {} Rep {} \n".format(task_id + 1, rep + 1))

            env = envs[task_id] # 현재 학습할 태스크 환경 설정
            if restart: # 이전 학습 이어서 진행
                start_step = restart_step # 이전 학습 스텝부터 시작
                restart = False
            else: # 새로운 태스크 학습 시작
                start_step = 0
            
            replay.set_task(task_id) # 현재 학습할 태스크를 replay buffer에 설정정
            step = common.Counter(start_step) # 시작 스텝 설정

            # 환경 전처리 적용
            env = common.GymWrapper(env)
            env = common.ResizeImage(env)
            if hasattr(env.act_space['action'], 'n'):
                env = common.OneHotAction(env)
            else:
                env = common.NormalizeAction(env)
            env = common.TimeLimit(env, config.time_limit)

            # 환경과 상호작용하면서 데이터를 수집하고, 여러 개의 환경을 병렬로 실행할 수 있도록 관리
            driver = common.Driver([env])
            driver.on_episode(per_episode) # 에피소드 종료시 로그 기록
            driver.on_step(lambda tran, worker: total_step.increment()) # 스텝이 수행될 때마다 총 스텝 수 증가
            driver.on_step(replay.add_step) # 스텝이 수행될 때마다 replay buffer에 데이터 추가
            driver.on_reset(replay.add_step) # 리셋될 때마다 replay buffer에 데이터 추가
            driver.on_step(lambda tran, worker: step.increment()) # 현재 태스크에서 진행 중인 스텝 수 증가

            # 모델 기반 탐색을 위한 초기 데이터 수집
            # replay buffer에 저장된 데이터가 설정된 prefill보다 적을 경우 데이터 수집
            prefill = max(0, config.prefill - replay.stats['total_steps'])
            if prefill:
                print(f'Prefill dataset ({prefill} steps).')
                random_agent = common.RandomAgent(env.act_space) # 무작위로 데이터 수집할 agent 생성
                driver(random_agent, steps=prefill, episodes=1) # 환경에서 무작위로 행동을 수행
                driver.reset() # 환경 초기화

            # agent 객체 생성 
            # obs_space: 관측 공간, act_space: 행동 공간, total_step: 학습된 스텝 수
            print('Create agent.')
            agnt = agent.Agent(config, env.obs_space, env.act_space, total_step)

            # Plan2Explore 탐색 전략을 사용하는 경우 agent에 설정
            if isinstance(agnt._expl_behavior, Plan2Explore):
                replay.agent = agnt

            # 데이터셋 준비 (배치 크기: 16, 길이: 50)
            dataset = iter(replay.dataset(**config.dataset)) # replay buffer에서 배치 크기 16, 길이 50으로 샘플링
            train_agent = common.CarryOverState(agnt.train) # 이전 상태를 유지하며 학습할 수 있도록 보장
            train_agent(next(dataset))  # 첫 번째 배치로 에이전트 학습
            
            # 이전 학습된 모델 가중치를 불러와 이어서 학습 -> 태스크 별로 구분하거나 이 부분을 임베딩 방식으로 해서 과거 데이터를 가져오는게 적절한 방법으로 보임임
            if (logdir / 'variables.pkl').exists(): # 파일이 존재하는지 확인
                print("Loading agent.")
                agnt.load(logdir / 'variables.pkl') # 파일이 존재하면 이전 학습된 모델 가중치를 불러옴
            else:
                # 사전 학습 수행 (지정된 pretrain 횟수만큼 반복)
                print('Pretrain agent.')
                for _ in range(config.pretrain): 
                    train_agent(next(dataset)) # 첫 번째 배치로 에이전트 학습

            # 정책 설정: 탐색 모드 또는 학습 모드 결정
            policy = lambda *args: agnt.policy(
                *args, mode='explore' if should_expl(total_step) else 'train') # 탐색 단계인지 학습 단계인지 판단
            

            # 평가 에피소드 수행 후 결과 기록하는 함수
            def eval_per_episode(ep, task_idx):
                length = len(ep['reward']) - 1 # 에피소드 길길이
                score = float(ep['reward'].astype(np.float64).sum()) # 에피소드 동안 받은 총 보상
                logger.scalar('eval_return_{}'.format(task_idx), score) # 태스크별 총 return 값 저장
                logger.scalar('eval_length_{}'.format(task_idx), length) # 태스크별 에피소드 길이 저장
                
                # 평가 결과를 비디오로 저장 (설정된 주기마다)
                if should_video_eval(total_step):
                    for key in config.log_keys_video:
                        logger.video(f'eval_{task_idx}_{total_step.value}', ep[key])
                
                # 재구성 오류 평가 (모델이 입력 데이터를 얼마나 잘 재구성하는지 확인)
                ep = {k: np.expand_dims(v, axis=0) for k, v in ep.items()} # 데이터를 차원 확장하여 모델이 처리할 수 있는 형태로 변환
                if should_recon(total_step): 
                    model_loss, _, _, _ = agnt.wm.loss(ep) # 월드 모델의 reconstruction loss 계산
                    logger.scalar('eval_recon_loss_{}'.format(task_idx), model_loss) # 재구성 오류 저장
                logger.write()

            # 에이전트가 리플레이 버퍼에서 데이터를 샘플링하여 학습 수행
            def train_step(tran, worker):
                if should_train(total_step):
                    for _ in range(config.train_steps):
                        mets = train_agent(next(dataset))  # 미니 배치를 가져와 에이전트 학습
                        [metrics[key].append(value) for key, value in mets.items()]
                
                # 설정된 주기마다 로깅 수행
                if should_log(total_step):
                    for name, values in metrics.items():
                        logger.scalar(name, np.array(values, np.float64).mean())
                        metrics[name].clear()
                    
                    # 비디오 저장 및 추가 로깅 수행
                    if should_video(total_step):
                        logger.add(agnt.report(next(dataset)))
                    logger.write(fps=True)

            # 각 스텝마다 학습 수행
            driver.on_step(train_step)

            # 평가 드라이버 생성 및 평가 실행
            eval_driver = common.Driver(_eval_envs, cl=config.cl)
            eval_driver.on_episode(eval_per_episode)  # 평가 에피소드가 끝날 때마다 실행
            
            # 평가 정책 설정, 학습 정책이 실제 환경에서 얼마나 성능이 좋은지 평가(일반적으로 학습 정책과 동일)
            # 탐색 없이 결정론적 방식으로 행동 수행
            # 평가 시에는 학습된 최적 행동을 수행하도록 설정
            eval_policy = lambda *args: agnt.policy(*args, mode='eval') # 탐색 없이 평가 모드로 정책 실행

            # 학습이 태스크마다 다른 스텝 수를 가질 경우, 각 태스크별로 학슥 스텝 수를 다르게 설정
            if unbalanced_steps is not None:
                steps_limit = int(unbalanced_steps[task_id]) # 태스크별로 지정된 학습 스텝을 가져옴
            else:
                steps_limit = config.steps # 기본적으로 각 태스크가 학습할 스텝 수

            while step < steps_limit: # 설정된 스텝 수만큼 학습 수행
                logger.write()
                # policy를 사용하여 현재 학습된 정책에 따라 환경에서 행동을 수행
                # eval_every는 평가 주기를 조절, 즉 일정한 스텝마다 에이전트가 환경과 상호작용하며 학습 진행
                driver(policy, steps=config.eval_every)
                
                # 탐색 및 평가 정책을 분리하는 경우 다르게 실행
                if config.sep_exp_eval_policies: # 탐색 정책과 평가 정책을 다르게 설정
                    eval_driver(eval_policy, steps=config.eval_steps)
                else: # 탐색 정책과 평가 정책을 동일하게 설정
                    eval_driver(policy, steps=config.eval_steps)
                
                # 학습된 모델의 가중치를 저장하여 이후 학습에 사용
                agnt.save(logdir / 'variables.pkl')

            # 다음 태스크로 이동
            task_id += 1

        # 학습 반복 수 증가
        rep += 1
