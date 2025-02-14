"""Plot evaluation results
"""

import argparse
import os
import plotly.graph_objs as go


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        Arguments
    """
    title = [None, "Arc_challenge", "Arc_easy", "Hellaswag", "winogrande"]
    # title=[None, "Total Score", "Format Correctness", "Grammatical Correctness", "State Consistency", "Contradiction Resolution", "Causal Clarity", "Information Relevance", "Tone Consistency"]

    base = [
        [1235814400],
        [35.49],
        [68.56],
        [45.15],
        [60.73],
    ]
    dp = [
        [1174992896, 992528384, 749242368, 566777856],
        [30.38, 24.83, 18.94, 17.83],
        [57.15, 48.32, 38.80, 35.98],
        [44.59, 38.47, 29.65, 27.34],
        [59.37, 58.88, 53.12, 52.01],
    ]
    wp_mlp = [
        [1112541184, 865306624, 618072064],
        [30.89, 27.56, 25.43],
        [59.81, 55.68, 52.74],
        [45.18, 42.6, 36.6],
        [59.67, 57.22, 55.8],
    ]
    wp_mlp_attn = [
        [1112508416, 865208320, 617908224, 371394560],
        [30.97, 29.86, 26.88, 21.5],
        [59.3, 60.02, 52.57, 42.63],
        [44.98, 41.87, 35.86, 29.33],
        [58.17, 55.8, 53.67, 50.59],
    ]
    wp_ch = [
        [1111423439, 864115261, 618270445, 370962267],
        [30.12, 23.38, 20.14, 19.8],
        [52.57, 46.3, 44.11, 43.06],
        [42.43, 31.54, 28.84, 27.82],
        [54.38, 50.12, 50.28, 52.8],
    ]
    wp_ch_mlp = [
        [1113555267, 865119593, 618389278, 371399085],
        [28.34, 26.54, 23.21, 20.73],
        [53.28, 50.04, 47.77, 43.77],
        [42.06, 37.49, 33.38, 28.64],
        [55.88, 52.8, 51.93, 51.93],
    ]
    wp_ch_mlp_attn = [
        [1112126323, 864092313, 617656507, 371219473],
        [27.39, 22.53, 20.05, 19.11],
        [55.47, 48.15, 46.13, 41.2],
        [42.04, 30.85, 29.07, 27.86],
        [52.72, 52.09, 53.12, 51.78],
    ]

    for y_idx in range(1, len(title)):
        fig = go.Figure(
            data=[
                go.Scatter(x=base[0], y=base[y_idx], mode="lines+markers", name="Base"),
                go.Scatter(
                    x=dp[0], y=dp[y_idx], mode="lines+markers", name="Depth Pruning"
                ),
                go.Scatter(
                    x=wp_mlp[0],
                    y=wp_mlp[y_idx],
                    mode="lines+markers",
                    name="Width Pruning (mlp)",
                ),
                go.Scatter(
                    x=wp_mlp_attn[0],
                    y=wp_mlp_attn[y_idx],
                    mode="lines+markers",
                    name="Width Pruning (mlp+attn)",
                ),
                go.Scatter(
                    x=wp_ch[0],
                    y=wp_ch[y_idx],
                    mode="lines+markers",
                    name="Width Pruning (ch)",
                ),
                go.Scatter(
                    x=wp_ch_mlp[0],
                    y=wp_ch_mlp[y_idx],
                    mode="lines+markers",
                    name="Width Pruning (ch+mlp)",
                ),
                go.Scatter(
                    x=wp_ch_mlp_attn[0],
                    y=wp_ch_mlp_attn[y_idx],
                    mode="lines+markers",
                    name="Width Pruning (ch+mlp+attn)",
                ),
            ],
        )

        fig.update_layout(
            title_text=title[y_idx],
            title_x=0.5,
            title_y=0.9,
            title_xanchor="center",
            title_yanchor="middle",
        )
        fig.update_xaxes(title_text="# Parameters")
        fig.update_yaxes(title_text="Score")
        fig.update_layout(margin=dict(l=0, r=0, b=0))

        fig.write_image(os.path.join(args.output_dir, f"{title[y_idx]}.png"))


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser(
        "Test script description",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for saving plot images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    os.makedirs(opts.output_dir, exist_ok=True)
    main(opts)
