# https://github.com/pytorch/rl
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import os
from torch import multiprocessing


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0) if torch.cuda.is_available() and not is_fork
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device("cpu")
)

torch.manual_seed(22)
env = GymEnv("Pendulum-v1", device=device) 
env.set_seed(22)
model = TensorDictModule(
        nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2),
            NormalParamExtractor()
            ),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
        ).to(device)

critic = ValueOperator(
        nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
            ),
        in_keys=["observation"],
        ).to(device)

model_filename = 'model.pt'
critic_filename = 'critic.pt'

if os.path.isfile(model_filename):
    model.load_state_dict(torch.load(model_filename))
if os.path.isfile(critic_filename):
    critic.load_state_dict(torch.load(critic_filename))

actor = ProbabilisticActor(
        model,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True
        )
buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(1000),
        sampler=SamplerWithoutReplacement(),
        batch_size=50,
        )
collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=1000,
        total_frames=1_000_000,
        )

loss_fn = ClipPPOLoss(actor, critic)
adv_fn = GAE(value_network=critic, average_gae=True, gamma=0.99, lmbda=0.95, device=device)
optim = torch.optim.Adam(loss_fn.parameters(), lr=2e-4)

for data in collector:  # collect data
    for epoch in range(10):
        adv_fn(data)  # compute advantage
        buffer.extend(data)
        for sample in buffer:  # consume data
            loss_vals = loss_fn(sample.to(device))
            loss_val = sum(
                    value for key, value in loss_vals.items() if
                    key.startswith("loss")
                    )
            loss_val.backward()
            optim.step()
            optim.zero_grad()
    print(f"avg reward: {data['next', 'reward'].mean().item(): 4.4f}")

torch.save(model.state_dict(), model_filename)
torch.save(critic.state_dict(), critic_filename)

#from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.envs import TransformedEnv

path = "./training_loop_ppo"
logger = CSVLogger(exp_name="PPO", log_dir=path, video_format="mp4")
video_recorder = VideoRecorder(logger, tag="video")
record_env = TransformedEnv(
        GymEnv("Pendulum-v1", from_pixels=True, pixels_only=False),
        video_recorder
        )

record_env.rollout(max_steps=2000, policy=actor.to('cpu'))
video_recorder.dump()
