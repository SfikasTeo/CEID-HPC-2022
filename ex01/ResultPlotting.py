import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

##########------------------   IMPORTANT   ------------------##########
# This script uses matplotlib and numpy in order to generate
# the wanted graphs from a specified .txt format. 
# Code is not optimized but we opted to generalize it as much as possible. 
# pip install -r requirements.txt may be needed.
# The extracted value is the elapsed time of each file ( second line ).
# IMPORTANT: the naming of the files format must be NAME.PROCS.TIMESTEPS.N.txt 
# The python executable must be at the same directory as the results of our code


###--- Append all txts from current Directory ---###
results = glob.glob('*.txt')

###--- Initialize a custom dataframe ---###

# There are certain cases for creating our dataframe ->
# 1. Consider that the number of processes(P) is changing while 
# the Timesteps (T) and the matrix dimensions (N) are constant.
# 2. The Timesteps (T) change while (P) and (N) remain constant.
# 3. The dimension (N) changes while (T) and (P) are stable.

const_N = sys.argv[1]
const_T = sys.argv[2]

# In our execution the number of processes is fixed to either 1 or 4.
# Our charts will illustrate cases 2 and 3.
dataframe_2 = []   # The constant (N) will be passed based on argv[1]
dataframe_3 = []   # The constant (T) will be passed based on argv[2]

# We need the number of different (N) and (T)
set_N = set()
set_T = set()
for each in results:
    partitioned = each.split(".")
    set_N.add(int(partitioned[3]))
    set_T.add(int(partitioned[2]))
set_N = list(sorted(set_N))
set_T = list(sorted(set_T))

# Initialzie Dataframe_3
for each in results:
    already_in_dataframe = False
    partitioned = each.split(".")

    # Check wether the txt belongs to dataframe_3 
    if partitioned[2] != const_T: continue
    for elements in dataframe_3:
        if ( (partitioned[0]+ "-" +partitioned[1]) == elements[0]):
            already_in_dataframe = True
            index = set_N.index(int(partitioned[3]))
            
            with open(each,"r") as file:
                listoflines = file.readlines()
                elements[index+1] = float(listoflines[1])
            file.close()
            break
    if already_in_dataframe == False:
        new_element = [0]*(len(set_N)+1)
        new_element[0] = partitioned[0] + "-" + partitioned[1]
        index = set_N.index(int(partitioned[3]))
        with open(each,"r") as file:
            listoflines = file.readlines()
            new_element[index+1] = float(listoflines[1])
        file.close()
        dataframe_3.append(new_element)

print(dataframe_3)

def plotResults_X( xaxis, inputlines, title, xlabel, ylabel, yaxis_collision_avertion ):
    ###--- Linestyles, colors and markers ---###
    
    #8 available line colors:
    mycolors = ["lightcoral","olive","seagreen","turquoise","steelblue","orchid","indigo","teal"] 
   	###--- create the plot ---###
       
    plt.rcParams['figure.figsize'] = [14, 14]
    plt.style.use("fivethirtyeight")

	# Create container figure object fig and axes.
    fig, ax = plt.subplots()
    
	# Add labels-title of ax
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.yaxis.labelpad=15

    # Set ticks of axis
    ax.set_xticks( xaxis )
    
    yaxis_with_collisions = []
    for each in inputlines:
        for value in each[1:]:
                yaxis_with_collisions.append( value )
    yaxis_with_collisions.sort()

    yaxis_without_collisions = [yaxis_with_collisions[0]]
    for each in yaxis_with_collisions[1:]:
        if abs(yaxis_without_collisions[-1] - each) > yaxis_collision_avertion:
            yaxis_without_collisions.append(each)

    ax.set_yticks( yaxis_without_collisions )
    ax.tick_params(axis='y', labelsize="11")

	###--- plot the input ---###
    # line is a tuple where line[0] is the index and line[1] is the list
    for line in enumerate(inputlines):
        iter_color = mycolors[ line[0] % len(mycolors) ]
        ax.plot( xaxis, line[1][1:], linewidth= 0.8, marker='.', markerfacecolor= iter_color,label= line[1][0], linestyle= "solid", color= iter_color )
       
    ax.legend(loc="upper left")

    #save the image to your working directory
    plt.savefig("{0}.png".format(title))
    plt.clf()


plotResults_X( set_N, dataframe_3, "Heat 2d Diffusion: Execution times", "Matrix Dimension (N)", "Time (s)", 1.5)