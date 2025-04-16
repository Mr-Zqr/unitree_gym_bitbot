import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import inspect

import argparse
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def cp_env(env, logdir):
    file_path = inspect.getfile(env.__class__)
    directory = os.path.dirname(file_path)
    print("copying env from ", directory, " to ", logdir)
    os.system(f"cp -r {directory} {logdir}")
    # 将directory最后一个//之间换成base
    os.system(f"cp -r {directory}/../base/ {logdir}")
    print("copying env from ", directory, " to ", logdir)
    # dir last name
    dir_name = os.path.basename(os.path.normpath(directory))
    os.system(f"rm -r {logdir}/{dir_name}/__pycache__")

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},
        {"name": "--show_log", "action": "store_true", "default": False, "help": "record log"},
        {"name": "--vel_debug", "action": "store_true", "default": False, "help": "show velocity in visualization"},
        {
            "name": "--backup_env",
            "action": "store_true",
            "default": True,
            "help": "Backup env and config files in the log directory",
        },
    ]
    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

# def export_policy_as_jit(actor_critic, path):
#     if hasattr(actor_critic, 'memory_a'):
#         # assumes LSTM: TODO add GRU
#         exporter = PolicyExporterLSTM(actor_critic)
#         exporter.export(path)
#     else: 
#         os.makedirs(path, exist_ok=True)
#         path = os.path.join(path, 'policy_1.pt')
#         model = copy.deepcopy(actor_critic.actor).to('cpu')
#         traced_script_module = torch.jit.script(model)
#         traced_script_module.save(path)
# def export_policy_as_jit(actor_critic, path, export_onnx=True):
#     """
#     Export policy model to specified formats.
    
#     Args:
#         actor_critic: The actor-critic model to export
#         path (str): Directory to save the models
#         export_onnx (bool): Whether to also export as ONNX format
    
#     Returns:
#         dict: Paths to the exported model files
#     """
#     exported_paths = {}
    
#     if hasattr(actor_critic, 'memory_a'):
#         lstm = actor_critic.memory_a.rnn
#         print(f"LSTM type: {type(lstm)}")
#         print(f"LSTM attributes: {dir(lstm)}")
#         print(f"LSTM parameters: {lstm}")
#         # LSTM or other recurrent network
#         exporter = PolicyExporterLSTM(actor_critic)
#         # Export to TorchScript (.pt)
#         pt_path = exporter.export(path)
#         exported_paths['torchscript'] = pt_path
        
#         # Export to ONNX if requested
#         if export_onnx:
#             onnx_path = exporter.export_onnx(path)
#             exported_paths['onnx'] = onnx_path
#     else:
#         # Non-recurrent network (MLP only)
#         os.makedirs(path, exist_ok=True)
#         pt_path = os.path.join(path, 'policy_1.pt')
#         model = copy.deepcopy(actor_critic.actor).to('cpu')
#         traced_script_module = torch.jit.script(model)
#         traced_script_module.save(pt_path)
#         exported_paths['torchscript'] = pt_path
        
#         # Export to ONNX if requested
#         if export_onnx:
#             onnx_path = os.path.join(path, 'policy_1.onnx')
#             dummy_input = torch.zeros((1, model.input_size), dtype=torch.float32)
#             torch.onnx.export(
#                 model=model,
#                 args=dummy_input,
#                 f=onnx_path,
#                 verbose=False,
#                 input_names=["observation"],
#                 output_names=["action"],
#                 dynamic_axes={
#                     'observation': {0: 'batch_size'},
#                     'action': {0: 'batch_size'}
#                 },
#                 do_constant_folding=True,
#                 opset_version=13,
#                 export_params=True
#             )
#             exported_paths['onnx'] = onnx_path
    
#     return exported_paths

def export_policy_as_jit(actor_critic, path, export_onnx=True):
    """
    Export policy model to specified formats.
    
    Args:
        actor_critic: The actor-critic model to export
        path (str): Directory to save the models
        export_onnx (bool): Whether to also export as ONNX format
    
    Returns:
        dict: Paths to the exported model files
    """
    exported_paths = {}
    
    if hasattr(actor_critic, 'memory_a'):
        # 获取模型输入大小
        try:
            input_size = actor_critic.memory_a.rnn.input_size
            print(f"Detected input size: {input_size}")
        except AttributeError:
            print("Could not detect input size directly, attempting to infer from model structure...")
            # 尝试从LSTM权重参数推断输入大小
            for name, param in actor_critic.memory_a.rnn.named_parameters():
                if 'weight_ih_l0' in name:
                    input_size = param.shape[1]
                    print(f"Inferred input size from weights: {input_size}")
                    break
            else:
                # 如果无法推断，使用默认值
                print("Warning: Could not infer input size, using default value 47")
                input_size = 47
        
        # LSTM模型
        exporter = PolicyExporterLSTM(actor_critic)
        exporter_pt = PolicyExporterLSTMPT(actor_critic)
        
        # 导出为TorchScript
        pt_path = exporter_pt.export(path)
        exported_paths['torchscript'] = pt_path
        
        # 导出为ONNX
        if export_onnx:
            onnx_path = exporter.export_onnx(path)
            exported_paths['onnx'] = onnx_path
            
            # 验证ONNX模型（可选）
            try:
                import onnx
                print("Validating ONNX model...")
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("ONNX model validation successful")
                
                # 尝试简化ONNX模型（如果安装了onnxsim）
                try:
                    import onnxsim
                    print("Simplifying ONNX model...")
                    model_simp, check = onnxsim.simplify(onnx_model)
                    if check:
                        print("ONNX model simplified successfully")
                        onnx.save(model_simp, onnx_path)
                    else:
                        print("ONNX model simplification failed")
                except ImportError:
                    print("onnxsim not installed, skipping simplification")
            except ImportError:
                print("onnx package not installed, skipping validation")
            except Exception as e:
                print(f"ONNX validation error: {e}")
    else:
        # 非LSTM模型（仅MLP）
        os.makedirs(path, exist_ok=True)
        pt_path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(pt_path)
        exported_paths['torchscript'] = pt_path
        
        # 导出为ONNX
        if export_onnx:
            onnx_path = os.path.join(path, 'policy_1.onnx')
            
            # 尝试获取输入大小
            input_size = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and input_size == 0:
                    input_size = module.in_features
                    break
            
            if input_size == 0:
                print("Warning: Could not determine input size for MLP, using default value 64")
                input_size = 64
                
            dummy_input = torch.zeros((1, input_size), dtype=torch.float32)
            torch.onnx.export(
                model=model,
                args=dummy_input,
                f=onnx_path,
                verbose=True,
                input_names=["observation"],
                output_names=["action"],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                },
                do_constant_folding=True,
                opset_version=11,
                export_params=True
            )
            exported_paths['onnx'] = onnx_path
    
    return exported_paths

class PolicyExporterLSTMPT(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    def export_onnx(self, path, input_size):
        """
        Exports the model to ONNX format.

        Args:
            path (str): Directory to save the ONNX file.
            input_size (tuple): The shape of the input tensor (e.g., (batch_size, input_dim)).
        """
        os.makedirs(path, exist_ok=True)
        onnx_path = os.path.join(path, 'policy_lstm.onnx')

        # Create dummy input for tracing
        dummy_input = torch.zeros(input_size, dtype=torch.float32).to("cuda:0")
        # dummy_hidden_state = torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32)
        # dummy_cell_state = torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32)

        # example = torch.ones((1, dummy_input)).to("cpu")
        torch.onnx.export(
            model=self.actor,
            args=dummy_input,
            f=onnx_path,
            verbose=False,
            input_names=["observation"],
            output_names=["action"],
            opset_version=17,
        )
        print(f"Model exported to ONNX format at {onnx_path}")

# class PolicyExporterLSTM(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.is_recurrent = actor_critic.is_recurrent
#         self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
#         self.memory.cpu()
#         self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
#         self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
#         self.input_size = self.memory.input_size # Store input size

#     def forward(self, x):
#         out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
#         self.hidden_state[:] = h
#         self.cell_state[:] = c
#         return self.actor(out.squeeze(0))

#     @torch.jit.export
#     def reset_memory(self):
#         self.hidden_state[:] = 0.
#         self.cell_state[:] = 0.
 
#     def export(self, path):
#         """Export model to TorchScript format (.pt)"""
#         os.makedirs(path, exist_ok=True)
#         pt_path = os.path.join(path, 'policy_lstm_1.pt')
#         self.to('cpu')
#         traced_script_module = torch.jit.script(self)
#         traced_script_module.save(pt_path)
#         print(f"Model exported to TorchScript format at {pt_path}")
#         return pt_path

#     def export_onnx(self, path):
#         """
#         Exports the LSTM+MLP model to ONNX format.
        
#         Args:
#             path (str): Directory to save the ONNX file.
#         """
#         os.makedirs(path, exist_ok=True)
#         onnx_path = os.path.join(path, 'policy_lstm_1.onnx')
        
#         # Move model to CPU for export
#         self.to('cpu')
        
#         # Create dummy input for tracing - use the expected input size
#         dummy_input = torch.zeros((1, self.input_size), dtype=torch.float32)
        
#         # Export the full LSTM+MLP model
#         torch.onnx.export(
#             model=self,
#             args=dummy_input,
#             f=onnx_path,
#             verbose=False,
#             input_names=["observation"],
#             output_names=["action"],
#             dynamic_axes={
#                 'observation': {0: 'batch_size'},
#                 'action': {0: 'batch_size'}
#             },
#             do_constant_folding=True,
#             opset_version=13,  # Using a compatible opset version
#             export_params=True
#         )
#         print(f"Model exported to ONNX format at {onnx_path}")
#         return onnx_path


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        
        # 获取LSTM的隐藏维度和层数
        self.hidden_size = self.memory.hidden_size
        self.num_layers = self.memory.num_layers
        
        # 获取输入大小 - 从LSTM的参数中推断
        self.input_size = self.memory.input_size if hasattr(self.memory, 'input_size') else 47  # 如果无法获取，使用默认值
        
        # 初始化隐藏状态和单元状态 - 但不再使用register_buffer
        self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.cell_state = torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, x, hidden_state=None, cell_state=None):
        """前向传播，接受隐藏状态和单元状态作为可选输入"""
        # 如果没有提供状态，使用默认状态
        if hidden_state is None:
            hidden_state = self.hidden_state
        if cell_state is None:
            cell_state = self.cell_state
            
        # LSTM前向传播
        out, (new_hidden, new_cell) = self.memory(x.unsqueeze(0), (hidden_state, cell_state))
        
        # Actor前向传播
        action = self.actor(out.squeeze(0))
        
        # 返回动作和新状态
        return action, new_hidden, new_cell

    def reset_memory(self):
        """重置LSTM状态"""
        self.hidden_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.cell_state = torch.zeros(self.num_layers, 1, self.hidden_size)
        
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    def export_onnx(self, path):
        """导出为ONNX格式，包含状态处理"""
        os.makedirs(path, exist_ok=True)
        onnx_path = os.path.join(path, 'policy_lstm_1.onnx')
        
        # 将模型移至CPU并设置为评估模式
        self.to('cpu')
        self.eval()
        
        # 创建用于导出的虚拟输入
        batch_size = 1
        dummy_input = torch.zeros((batch_size, self.input_size), dtype=torch.float32)
        dummy_hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float32)
        dummy_cell = torch.zeros((self.num_layers, batch_size, self.hidden_size), dtype=torch.float32)
        
        # 定义输入和输出名称
        input_names = ["observation", "hidden_state", "cell_state"]
        output_names = ["action", "new_hidden_state", "new_cell_state"]
        
        # 定义动态轴
        # dynamic_axes = {
        #     'observation': {0: 'batch_size'},
        #     'hidden_state': {1: 'batch_size'},
        #     'cell_state': {1: 'batch_size'},
        #     'action': {0: 'batch_size'},
        #     'new_hidden_state': {1: 'batch_size'},
        #     'new_cell_state': {1: 'batch_size'}
        # }
        
        # 导出模型
        with torch.no_grad():
            torch.onnx.export(
                model=self,
                args=(dummy_input, dummy_hidden, dummy_cell),  # 传递所有输入参数
                f=onnx_path,
                verbose=True,
                input_names=input_names,
                output_names=output_names,
                # dynamic_axes=dynamic_axes,
                opset_version=11,  # 使用较低版本以保证兼容性
                do_constant_folding=True,
                export_params=True
            )
        
        print(f"Model exported to ONNX format at {onnx_path}")
        return onnx_path

# overide gymutil
def parse_device_str(device_str):
    # defaults
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id

def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    if args.device is not None:
        args.sim_device = args.device
        args.rl_device = args.device
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args