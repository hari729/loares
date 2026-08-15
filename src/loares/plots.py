from pymoo.visualization.scatter import Scatter
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.pcp import PCP


def save_scatter_plots(F, spec, path):
    if spec["problem"].n_obj <= 3:
        plot = Scatter(legend=True)
        plot.add(F, label=spec["algorithm_name"], **spec["plot_kwargs"])
        plot.save(path)
    else:
        plot = PCP()
        plot.add(F)
        plot.save(path)


def save_heatmap(F, x_labels, y_labels, path):
    plot = Heatmap(
        bounds=[0, 1],
        title=("Optimization", {"pad": 15}),
        cmap="Oranges_r",
        solution_labels=y_labels,
        labels=x_labels,
    )
    plot.add(F)
    plot.save(path)
