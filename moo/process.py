import matplotlib.pyplot as plt
from opti.core.process import ResultProcessor
from opti.core.problem import Problem
from opti.moo.population import MoPopulation

class MOProcessor(ResultProcessor):
    def __init__(self,
                 results_list,
                 root_dir = None):
        super().__init__(results_list,
                         root_dir)

    def plot_convergence(self, convergence_data, legend, file_path):
        colors = {
            "GD" : "red",
            "IGD" : "blue",
            "SPC" : "green",
            "SPR" : "orange",
            "HV" : "indigo"
        }
        for key in convergence_data:
            if key == "evals":
                continue

            plt.figure()
            plt.plot(convergence_data["evals"], convergence_data[key], linestyle='-',marker='',
                    color=colors[key],
                    markerfacecolor='cyan',markersize='5',
                    markeredgecolor='black',markeredgewidth=0.1)
            plt.legend(labels=legend, loc='right', fontsize=8)
            plt.grid(which='both',linestyle='--',alpha=0.7)
            plt.xlabel("Function Evaluations")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(f"{file_path}/{key}.png", dpi=600, bbox_inches='tight')
            plt.close()

    def plot_pareto_front(self, objective_values, tf, legend, file_path):
        n_obj = objective_values.shape[1]
        if n_obj == 2:
            plt.figure()
            plt.plot(objective_values[:,0], objective_values[:,1], linestyle='',marker='s',
                    markerfacecolor='cyan',markersize='5'
                    ,markeredgecolor='black',markeredgewidth=0.1)
            if tf is not None:
                plt.plot(tf[:,0],tf[:,1],linestyle='',marker='.',color='black'
                        ,markersize='5',alpha=1)
                legend.append("True Front")
            plt.legend(labels=legend, loc='upper right', fontsize=8)
            plt.grid(which='both',linestyle='--',alpha=0.7)
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.tight_layout()
            plt.savefig(f"{file_path}/pareto-front.png", dpi=600, bbox_inches='tight')
            plt.close()
        
        if n_obj == 3:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(elev=30, azim=30)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            
            plt.plot(objective_values[:,0], objective_values[:,1],objective_values[:,2], linestyle='',marker='s',
                        markerfacecolor='cyan',markersize='5',markeredgecolor='black',markeredgewidth=0.1)
            if tf is not None:
                plt.plot(tf[:,0],tf[:,1],tf[:,2],linestyle='',marker='.',color='black',markersize='5')
                legend.append("True Front")
            plt.legend(labels=legend, loc='upper right', fontsize=8)
            ax.grid(which='both',linestyle='--',alpha=0.3)
            plt.savefig(f"{file_path}/pareto-front.png", dpi=600, bbox_inches='tight')
            plt.close()

    def generate_plots(self, result, save_path):
        _,pareto_front,_,_ = result.population.get_pareto()
        legend = [result.algorithm.get_info()["name"]]
        self.plot_convergence(result.get_convergence_data(), legend, save_path)
        self.plot_pareto_front(pareto_front,
                                result.problem.get_true_front(),
                                legend,
                                save_path)





