import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils.utils import get_action, save_checkpoint
from collections import deque
from utils.running_state import ZFilter
from hparams import HyperParams as hp
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG,'
                         'NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Hopper-v3",
                    help='name of Mujoco environement')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
parser.add_argument('--seed', type=int, default=500)
args = parser.parse_args()

if args.algorithm == "PG":
    from agent.vanila_pg import train_model
elif args.algorithm == "NPG":
    from agent.tnpg import train_model
elif args.algorithm == "TRPO":
    from agent.trpo_gae import train_model
elif args.algorithm == "PPO":
    from agent.ppo_gae import train_model


if __name__=="__main__":
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    exp_id = args.algorithm + '_' + args.env + '_seed_' + str(args.seed)
    args.logdir = os.path.join(args.logdir, exp_id)
    model_path = os.path.join(os.getcwd(),'save_model', exp_id)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    writer = SummaryWriter(args.logdir)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    running_state = ZFilter((num_inputs,), clip=5)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr,
                              weight_decay=hp.l2_rate)

    episodes = 0
    for iter in range(15000):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        while steps < 2048:
            episodes += 1
            state = env.reset()
            state = running_state(state)
            score = 0
            for _ in range(10000):
                if args.render:
                    env.render()

                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break

        # evaluation:
        steps = 0
        scores = []
        while steps < 2048:
            state = env.reset()
            state = running_state(state)
            score = 0
            for _ in range(10000):
                if args.render:
                    env.render()
                
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = mu.detach().numpy()[0]
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                score += reward
                state = next_state

                if done:
                    break
            
            scores.append(score)
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

        if iter % 10000:
            score_avg = int(score_avg)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)
