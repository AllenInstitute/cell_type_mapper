import argparse

from cell_type_mapper.diff_exp.p_value_mask import (
    create_p_value_mask_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed_path', type=str,
        default='data/precomputed_stats.20231106.sea_ad.h5')
    parser.add_argument('--output_path', type=str,
        default='output/p_values.h5')
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    create_p_value_mask_file(
        precomputed_stats_path=args.precomputed_path,
        dst_path=args.output_path,
        n_processors=args.n_processors)

if __name__ == "__main__":
    main()
