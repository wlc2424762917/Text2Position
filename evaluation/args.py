import argparse
from argparse import ArgumentParser
import os
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Pos models and ablations")

    parser.add_argument("--purpose", type=str)

    # Paths
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="K360")
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--path_coarse", type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument(
        "--path_fine", type=str, help="The path to the Hints-to-Objects matching model"
    )

    # Options
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--threshs", type=int, nargs="+", default=[5, 10, 15]
    )  # Possibly over-write here when it becomes a list of tuples
    parser.add_argument("--pad_size", type=int, default=16)

    parser.add_argument(
        "--use_test_set", action="store_true", help="Run test-set instead of the validation set."
    )

    parser.add_argument("--no_pc_augment", action="store_true")
    parser.add_argument("--num_mentioned", type=int, default=6)

    parser.add_argument(
        "--plot_retrievals",
        action="store_true",
        help="Plot 3 success and fail examples, then quit.",
    )
    parser.add_argument("--plot_matches", action="store_true")
    parser.add_argument("--coarse_only", action="store_true")

    # Oracles
    parser.add_argument("--coarse_oracle", action="store_true", help="Use gt-retrievals")
    parser.add_argument(
        "--street_oracle", action="store_true", help="Use street-based coarse semi-oralce"
    )
    parser.add_argument("--coarse_random", action="store_true", help="Use random coarse retrieval")
    parser.add_argument("--fine_oracle", action="store_true", help="Use perfect in-cell locations")
    parser.add_argument("--fine_random", action="store_true", help="Use random in-cell locations")

    # Object-encoder / PointNet
    parser.add_argument("--pointnet_numpoints", type=int, default=256)

    # Various - don't change these!
    parser.add_argument("--ranking_loss", type=str, default="pairwise")

    parser.add_argument("--use_alt_descriptions", action="store_true")

    # no gt instance
    parser.add_argument("--no_objects", action="store_true")

    # fine stage
    parser.add_argument("--only_matcher", action="store_true")
    parser.add_argument("--use_cross_attention", action="store_true")
    parser.add_argument("--no_superglue", action="store_true")

    # mask3d part
    parser.add_argument("--mask3d_ckpt", type=str, default="/home/planner/wlc/Mask3D/saved/t2p_q24_015_56.2/last.ckpt")
    parser.add_argument("--freeze_cell_encoder", action="store_true")
    parser.add_argument("--use_queries", action="store_true")


    # Model
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument(
        "--lr_idx",
        default=None,
        type=int,
        help="Which learning rate index to use from a given fixed index. Only used to schedule multiple LRs at once, overwrites learning_rate if set.",
    )
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")

    parser.add_argument(
        "--continue_path", type=str, help="Set to continue from a previous checkpoint"
    )

    parser.add_argument("--no_cell_augment", action="store_true")

    # SuperGlue
    parser.add_argument("--sinkhorn_iters", type=int, default=50)
    parser.add_argument("--describe_by", type=str, default="all")

    # Cell retrieval
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5

    # Object-encoder / PointNet
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument(
        "--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth"
    )
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)

    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--num_embed", action="store_true")

    # PyG
    parser.add_argument("--use_edge_conv", action="store_true")

    # Relation Transformer
    parser.add_argument("--use_relation_transformer", action="store_true")

    # use clip feature as instance semantic feature
    parser.add_argument("--only_clip_semantic_feature", action="store_true")
    parser.add_argument("--use_clip_semantic_feature", action="store_true")

    # language encoder
    parser.add_argument("--language_encoder", type=str, default="LSTM")
    # parser.add_argument("--T5_model_path", type=str, default="/home/wanglichao/t5-small")
    parser.add_argument("--T5_model_path", type=str, default="/home/planner/wlc/t5-small")

    # Arguments for translation regressor training.
    parser.add_argument(
        "--regressor_dim",
        type=int,
        default=128,
        help="Embedding dimension of language encode before regressing offsets.",
    )
    # Variations which tranlations are fed into the network for training/evaluation.
    # NOTE: These variations did not make much difference.
    parser.add_argument("--regressor_cell", type=str, default="pose")  # Pose or best
    parser.add_argument("--regressor_learn", type=str, default="center")  # Center or closest
    parser.add_argument("--regressor_eval", type=str, default="center")  # Center or closest

    # unstable
    # use semantic head
    parser.add_argument("--use_semantic_head", action="store_true")
    parser.add_argument("--no_offset_idx", action="store_true")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--text_freeze", type=str2bool, nargs='?', const=True, default=True,
                        help="Activate clip_text_freeze")
    args = parser.parse_args()

    assert osp.isfile(args.path_coarse)
    assert osp.isfile(args.path_fine)
    if args.coarse_oracle:
        assert max(args.top_k) == 1, "Coarse oracle can only retrieved one best cell."

    if args.coarse_random:
        assert not args.coarse_oracle and not args.street_oracle

    if args.fine_random:
        assert not args.coarse_oracle and not args.fine_oracle

    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
