from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_29dof_config import G1Dof29RoughCfg, G1Dof29RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.tiangong.tg_config import TGConfig, TGConfigPPO
from legged_gym.envs.tiangong.tg_env import TGEnv
# from legged_gym.envs.hhfc.hhfc_config import HhfcConfig, HhfcConfigPPO
# from legged_gym.envs.hhfc.hhfc_env import HhfcEnv
from legged_gym.envs.bhr8.bhr8_config import BHR8RoughCfg, BHR8RoughCfgPPO
from legged_gym.envs.bhr8.bhr8_env import BHR8Robot
from legged_gym.envs.miniwheel.miniwheel_config import MiniWheelRoughCfg, MiniWheelRoughCfgPPO
from legged_gym.envs.miniwheel.miniwheel_env import MiniWheelRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "bhr8", BHR8Robot, BHR8RoughCfg(), BHR8RoughCfgPPO())
task_registry.register( "tiangong", TGEnv, TGConfig(), TGConfigPPO())
# task_registry.register( "hhfc", HhfcEnv, HhfcConfig(), HhfcConfigPPO())
task_registry.register( "miniwheel", MiniWheelRobot, MiniWheelRoughCfg(), MiniWheelRoughCfgPPO())