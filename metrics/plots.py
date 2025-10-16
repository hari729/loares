import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
# from matplotlib.backends.backend_pdf import PdfPages

def generate_plots_notf(function_name,algo_name,psize,max_evals,objective_values,legend,file_path,tf=None):

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
        plt.savefig(f"{file_path}/{function_name}_{algo_name}_{psize}_{max_evals}.png", dpi=600, bbox_inches='tight')
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
        plt.savefig(f"{file_path}/{function_name}_{algo_name}_{psize}_{max_evals}.png", dpi=600, bbox_inches='tight')
        plt.close()

def plot_mod(csv_path,legend,algo_name,cpts=None,xylabels=None, lloc=None):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(csv_path)
    # Extract as numpy array (without headers)
    objective_values = df.to_numpy()

    file_path = Path(csv_path)
    directory = file_path.parent

    n_obj = objective_values.shape[1]

    if xylabels is None:
        xylabels = [f"f{i+1}" for i in range(n_obj)]

    if n_obj == 2:
        plt.figure()
        plt.plot(objective_values[:,0], objective_values[:,1], linestyle='',marker='o',
                  markerfacecolor='blue',markersize='5',markeredgecolor='black',markeredgewidth=0.1)
        if cpts is not None:
            plt.plot(cpts[:,0],cpts[:,1],linestyle='',marker='x',color='red'
                    ,markersize='5',alpha=1)
        if lloc is None:
            plt.legend(labels=legend, loc='upper right', fontsize=8)
        else:
            plt.legend(labels=legend, loc=lloc, fontsize=8)
        plt.grid(which='both',linestyle='--',alpha=0.7)
        plt.xlabel(xylabels[0])
        plt.ylabel(xylabels[1])
        plt.tight_layout()
        plt.savefig(f"{directory}/mod_{algo_name}_{timestamp}.png", dpi=600, bbox_inches='tight')
        plt.close()
    
    if n_obj == 3:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=30, azim=30)
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        ax.set_zlabel(xylabels[2])
        
        plt.plot(objective_values[:,0], objective_values[:,1],objective_values[:,2], linestyle='',marker='s',
                    markerfacecolor='cyan',markersize='5',markeredgecolor='black',markeredgewidth=0.1)
        
        plt.legend(labels=legend, loc='upper left', fontsize=8)
        ax.grid(which='both',linestyle='--',alpha=0.3)
        plt.savefig(f"{file_path}/{function_name}_{algo_name}_{psize}_{iterations}.png", dpi=600, bbox_inches='tight')
        plt.close()

def convergence_plots(function_name,algo_name,psize,max_evals,convergence_data,file_path):
    n_metrics = convergence_data.shape[1]-1
    # print(n_metrics)
    if n_metrics == 5:
        ylabels = ["GD","IGD","SPC","SPR","HV"]
        colors = ["red","blue","green","orange","indigo"]
    elif n_metrics == 2:
        ylabels = ["SPC","HV"]
        colors = ["green","indigo"]

    legend = [f"MO-{algo_name.upper()}"]

    for i in range(0,n_metrics):

        plt.figure()
        plt.plot(convergence_data[:,-1], convergence_data[:,i], linestyle='-',marker='',
                 color=colors[i],
                  markerfacecolor='cyan',markersize='5',
                  markeredgecolor='black',markeredgewidth=0.1)
        plt.legend(labels=legend, loc='upper right', fontsize=8)
        plt.grid(which='both',linestyle='--',alpha=0.7)
        plt.xlabel("Function Evaluations")
        plt.ylabel(ylabels[i])
        plt.tight_layout()
        plt.savefig(f"{file_path}/{ylabels[i]}_{function_name}_{algo_name}_{psize}_{max_evals}.png", dpi=600, bbox_inches='tight')
        plt.close()


    
