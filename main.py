import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.api.environment import Environment
from torch.api.pipeline_spec import ModelSpec
from torch.core.config import Conf

import const
import torch
import data
# import hooks
import argparse
from lazydecoder import *
from model import *

import os
from torch.api.warm_start import *


class NNWarmStartSettings(LarcNNWarmStartSettings):
    def get_var_to_warm_start_rules(self):
        var_map = {}
        return var_map


parser = argparse.ArgumentParser(description="PyTorch Training Config")

parser.add_argument("--job_dtm", type=str, default="", help="model job dtm")
parser.add_argument("--model_config_path", type=str, default="config/model.yaml", help="model structure conf")
parser.add_argument("--warm_start_vars", type=str, default="", help="warm start model weight")
parser.add_argument("--trans_map_path", type=str, default="", help="trans_map_path")
parser.add_argument("--trans_info_path", type=str, default="", help="trans_info_path")
parser.add_argument("--serving_head", type=str, default="label", help="serving head name")
parser.add_argument("--warm_start_head", type=str, default="label", help="warm start head name")

parser.add_argument("--is_dump", action="store_true", help="whether to dump intermediate results")
parser.add_argument("--is_warm", action="store_true", help="whether to enable warm start")
parser.add_argument("--use_lhuc", action="store_true", help="whether to use LHUC adaptation")
parser.add_argument("--dump_hour_list", type=str, default="", help="list of dump hours")
parser.add_argument("--comp_idx_cut", type=int, default=0, help="comp index cutoff")

parser.add_argument("--alpha", type=float, default=0.75, help="alpha value")
parser.add_argument("--beta", type=float, default=1.0, help="beta value")
parser.add_argument("--gamma", type=float, default=2.0, help="gamma value")

parser.add_argument("--use_win_flow", action="store_true", help="whether to use win flow")
parser.add_argument("--sec_flow_ratio", type=float, default=1.0, help="second flow ratio")
parser.add_argument("--imp_weight", type=float, default=1.0, help="importance weight")
parser.add_argument("--is_uwl", action="store_true", help="whether to use uncertain weight loss")

parser.add_argument("--train_neg_sample_bias", action="store_true", help="enable bias in training negative sampling")
parser.add_argument("--eval_neg_sample_bias", action="store_true", help="enable bias in eval negative sampling")
parser.add_argument("--neg_sample_ratio", type=float, default=1.0, help="negative sample ratio")

parser.add_argument("--creativity_max_image_candidates", type=int, default=30, help="max image candidates")
parser.add_argument("--creativity_max_title_candidates", type=int, default=20, help="max title candidates")
parser.add_argument("--image_output_nums", type=int, default=30, help="number of image outputs")
parser.add_argument("--image_title_output_nums", type=int, default=30, help="number of image+title outputs")

parser.add_argument("--enable_lds_lastN", action="store_true", help="whether to enable LDS lastN")
parser.add_argument("--lds_file", type=str, default="core/conf/lds_lastN2.txt", help="LDS configuration file")

parser.add_argument("--creativity_image_emb_dim", type=int, default=64, help="embedding dim for image")
parser.add_argument("--creativity_title_emb_dim", type=int, default=16, help="embedding dim for title")

parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning rate for GRM/LazyDecoder")
parser.add_argument("--use_gbpo", action="store_true", help="enable RL stage (GBPO) during eval phase")
parser.add_argument("--lambda_rl", type=float, default=0.1, help="GBPO RL loss weight")
parser.add_argument("--clip_ratio", type=float, default=0.2, help="GBPO clip ratio")

FLAGS, unknown = parser.parse_known_args()


def mock_larc_env():
    sys.argv = [
        sys.argv[0],
        "--task_type", "chief",
        "--larc_report_pod_type", "cpu",
        "--larc_report_pod_id", "0",
        "--enable_ddp"  
    ]


def main():
    mock_larc_env() 
    mock_config()
    env = Environment(sys.argv)
    MODEL = Model(FLAGS.model_config_path, FLAGS.config_path, FLAGS.sparse_config_path)
    Conf.sub_graph_tag = None
    Conf.use_gbpo = FLAGS.use_gbpo

    train_filter_fn = MODEL.data_config.get('train_filter_fn', None)

    if FLAGS.enable_lds_lastN:
        const.init_lasLastN_feas(FLAGS.lds_file)

    filter_fn = None
    data_spec = data.ExportCacheParquetDataSpec(
        preprocess_fn=data.preprocess,
        filter_fn=filter_fn,
        raw_features_to_keep={
            "cacheImageVec", "cacheTitleVec", "label", "image_label", "click", "labelWeight",
            "trackId", "userId", "adsNoteId", "hasClick", "hasImpress",
            "image_emb_v1", "winImageId", "winTitleId", "image_token_list_v1",
            "keywords_v1", "people_tags_v1", "rank_score_v1", "useful_score_v1",
            "optimizeTarget"
        }
    )
    env.set_data_spec(data_spec)
    grm_model = GRM(
        model=MODEL,
        serving_head=FLAGS.serving_head,
        warm_start_head=FLAGS.warm_start_head
    )
    grm_model.optimizer = optim.Adam(grm_model.model.parameters(), lr=FLAGS.learning_rate)
    grm_model.trainer = GBPOTrainer(grm_model.model, lambda_rl=FLAGS.lambda_rl, clip_ratio=FLAGS.clip_ratio)

    env.set_estimator(grm_model)

    if FLAGS.is_warm:
        env.init()
        skip_table = ["dim-8-title"]
        ws_setting = NNWarmStartSettings(
            new_sparse_config_path="conf/sparse.yaml",
            old_sparse_config_path="conf/old_sparse.yaml"
        )
        env.load_model(
            ModelSpec(
                sparse_map_fn=(lambda x: None if x in skip_table else x),
                nn_warm_start_setting=ws_setting
            )
        )
    else:
        env.init_and_load_model()
    env.train()



def mock_config():
    param = FLAGS
    param.sparse_config_path = "./conf/sparse.yaml"
    param.config_path = "./conf/all.json"
    param.model_config_path = "./model.yaml"
    param.train_data_path = '/home/*.parquet'
    param.train_data_end_dtm = '20201020'
    param.train_epochs = 1
    param.eval_data_path = param.train_data_path.split(',')[0]
    param.core_model_dir = './model'
    param.dataset_read_buffer_size = 2621448
    param.dataset_batch_size = 256
    param.dataset_prefetch_buffer_size = 256
    param.dataset_num_parallel_calls = 2
    param.train_max_steps = 1000
    param.train_data_num_days = 1
    param.train_data_end_dtm = "20201020"
    param.train_data_dtm_format = "%Y%m%d"
    param.train_data_rounds = 1
    param.so_file_path = ''
    # param.trans_sparse_table_name = ['spn_oss_trans']
    param.is_local_debug = True
    param.dataset_interleave_cycle_length = 1
    param.save_checkpoints_steps = 100
    # param.is_trans=True
    # param.tower_tag=["finalAdsUser", "finalAdsNote", "finalAdsAuthorId"]
    # param.trans_model_group =  ["x2a_Larc_Training"]
    # param.trans_model_type=['finalAds']
    # param.trans_map_path="alc/data/sparse.yaml.finalAdsmapinfo"
    # param.trans_info_path="alc/data/sparse.yaml.finalAdsinfo"
    # param.multi_task_models = ['label', 'image_label']
    param.model_weight_boost = ['label:1', 'image_label:2']

    param.learning_rate = getattr(param, "learning_rate", 2e-5)
    param.use_gbpo = getattr(param, "use_gbpo", False)
    param.lambda_rl = getattr(param, "lambda_rl", 0.1)
    param.clip_ratio = getattr(param, "clip_ratio", 0.2)
    param.enable_ddp = True

if __name__ == '__main__':
    main()  