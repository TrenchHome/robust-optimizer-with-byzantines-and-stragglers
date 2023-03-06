from argparse import ArgumentParser
from grid_search import grid_search
from base_training import base_training

# -------------------------- Argument Parsing --------------------------
parser = ArgumentParser(description="")
parser.add_argument("--cuda", action="store_true")  # default = False
parser.add_argument("--debug", action="store_true") # default = False
parser.add_argument("--dont-save", action="store_true") # default = False
parser.add_argument("--dont-evaluate", action="store_true") # default = False
parser.add_argument("--dir", type=str, default="default")
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--lr", nargs="+", type=float)
parser.add_argument("--seed", nargs="+", type=int)
parser.add_argument("--regu", nargs="+", type=float)
parser.add_argument("--S", nargs="+", type=int)
parser.add_argument("--H", type=int, default=5)
parser.add_argument("--B", type=int, default=5)
parser.add_argument("--b", nargs="+", type=int)
parser.add_argument("--acc", action="store_true") # default = False
parser.add_argument("--loss", action="store_true") # default = False
parser.add_argument("--net", type=str)
parser.add_argument("--norm", action="store_true") # default = False
parser.add_argument("--p", nargs="+", type=float) # default = 1 # active probability
parser.add_argument("--nu", nargs="+", type=float) # reuploading param
parser.add_argument("--initfp", action="store_true") # default = False
parser.add_argument("--stopwhen", type=str, default="EXP") # default = EXP, choice = CM
parser.add_argument("--logcost", action="store_true") # default = EXP, choice = CM
parser.add_argument("--atk", nargs="+", type=str) # attack classes
parser.add_argument("--theo", action="store_true") # is the reuploading follows theo-value; default = False
parser.add_argument("--dataset", type=str) # dataset (MNIST or CIFAR10)
parser.add_argument("--weak-strag", action="store_true") # is the reuploading follows theo-value; default = False

args = parser.parse_args()

print(f"{args.stopwhen=}")

# file writing
IS_DEBUG = args.debug
IS_SAVING_FILE = not(args.dont_save)
IS_EVALUATION = not(args.dont_evaluate)
DIRNAME_SUFFIX = "({})".format(args.dir)
is_observe_variance = False
is_observe_weight = False
# gpu
USE_CUDA = args.cuda

# -------------------------- DATASET INFOS --------------------------
DATASET = args.dataset

REGU_COEFS = [0.0] if (args.regu is None) else args.regu
BATCH_SIZES = [32] if (args.b is None) else args.b

# -------------------------- TRAINING INFOS --------------------------
LR_LIST = args.lr

CRITERION_NAME = "CE" 
N_EPOCHS = args.epoch
N_BATCHES = 1 
LOG_INTERVAL = 1 

EVALUATION_INTERVAL = 1 if (DATASET == "MNIST") else 100

# -------------------------- CLIENT INFOS --------------------------
N_HON = args.H
N_BYZ = args.B
N_STRAG_LIST = args.S
# -------------------------- ATTACK INFOS --------------------------
NA_ATK = [
    { "atk_name": "NoAttack",   "scaling": None }, # v0
]

MNIST_ATKS = [
    { "atk_name": "ALIE",       "scaling": 1 },  # v0
    { "atk_name": "IPM",       "scaling": 100 },  # v0
    { "atk_name": "IPM_G",       "scaling": 0.1 },  # v0
    { "atk_name": "SM",        "scaling": None },
    { "atk_name": "BF",         "scaling": 1 }, # v0
    { "atk_name": "IPM_G",       "scaling": 1000 },  # v0
    { "atk_name": "IPM_G",       "scaling": 0.75 },  # v0
]

CIFAR_ATKS = [
    { "atk_name": "ALIE",       "scaling": 1000 },  # v0
    { "atk_name": "IPM",       "scaling": 1000 },  # v0
    { "atk_name": "IPM_G",       "scaling": 0.01 },  # v0
    { "atk_name": "IPM_G",       "scaling": 1000 },  # v0
    { "atk_name": "SM",        "scaling": None },
    { "atk_name": "BF",         "scaling": 1 }, # v0
]

NON_OMNI_ATKS = [
    { "atk_name": "RandGrad",    "scaling": 1 }, # v0
    { "atk_name": "GradShift",    "scaling": 5000 }, # v0
]

# ----------------------------------------------------

ATK_DICT = {
    'NA': NA_ATK,
    'NON_OMNI': NON_OMNI_ATKS,
    'MNIST': MNIST_ATKS,
    'CIFAR': CIFAR_ATKS,
}

ATK_INFOS = []
for atk_class in args.atk:
    ATK_INFOS += ATK_DICT[atk_class]
    
# -------------------------- DEFENSE INFOS --------------------------
AGG_INFOS = [
    {'agg_name': 'Avg',   'is_agg_HA': False},    
    {'agg_name': 'TM',  'is_agg_HA': False,   'n_byz': N_BYZ },  
    {'agg_name': 'CM',   'is_agg_HA': False},    
    {'agg_name': 'Krum',   'is_agg_HA': False, 'n_byz': N_BYZ, 'm': 1},   
    {'agg_name': 'RFA',   'is_agg_HA': False, 'n_byz': N_BYZ, 'T': 3, 'v': 0.1},   
    {'agg_name': 'CC',   'is_agg_HA': False, 'tau': 100, 'n_iter': 3},   
    {'agg_name': 'MDA',  'is_agg_HA': False,   'n_byz': N_BYZ },  
]
MMT_INFOS = [
    {"type": "client", "coef": 0.9, "gamma": 0.1 },    
    {"type": "client", "coef": 0.43, "gamma": 0.57 },  
    {"type": "client", "coef": 0, "gamma": 1 },  
    {"type": "client", "coef": 0.7, "gamma": 0.3 },  
    {"type": "client", "coef": 0.5, "gamma": 0.5 },
    {"type": "client", "coef": 0.3, "gamma": 0.7 },    
    {"type": "client", "coef": 0.1, "gamma": 0.9 },    
]

FILL_TYPES = [     
    "NoFill", 
    "Previous", 
    "Consensus", 
]
# -------------------------- RANDOM SEED --------------------------
SEEDS = [0] if (args.seed is None) else args.seed

# -------------------------- p_list and nu_list --------------------------
p_list = [None] if (args.p is None) else args.p
nu_list = [None] if (args.nu is None) else args.nu

# -------------------------- START GRID-SEARCH-ING --------------------------
grid_search(
    dataset=DATASET, network_name=args.net, is_normalized=args.norm,
    synthetic_info=None, criterion_name=CRITERION_NAME, 
    n_epochs=N_EPOCHS, n_batches=N_BATCHES, log_interval=LOG_INTERVAL, evaluation_interval=EVALUATION_INTERVAL,
    n_hon = N_HON, n_byz = N_BYZ, 
    seeds = SEEDS, lr_list = LR_LIST,
    regu_coefs = REGU_COEFS, batch_sizes = BATCH_SIZES, 
    n_strag_list = N_STRAG_LIST, 
    atk_infos = ATK_INFOS, agg_infos = AGG_INFOS, mmt_infos = MMT_INFOS, fill_types = FILL_TYPES,
    use_cuda = USE_CUDA, is_debug = IS_DEBUG, is_saving_file = IS_SAVING_FILE, dir_name_suffix = DIRNAME_SUFFIX,
    is_acc_required = args.acc, is_loss_required = args.loss,
    is_observe_variance = is_observe_variance, is_observe_weight = is_observe_weight, 
    is_log_weight = False, is_evaluation = IS_EVALUATION,
    p_list=p_list, nu_list=nu_list, 
    is_init_fp=args.initfp, stop_criterion_for_reupload=args.stopwhen, is_count_log_cost=args.logcost,
    is_theo_defense=args.theo, 
    is_weak_strag=args.weak_strag,
    fn = base_training
)
