import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_exp import *


if __name__ == '__main__' :

    # load data, split into test and train set, look at general properties of data
    data = import_data( 'data/Task_Data_Scientist_Dataset.xlsx' )

    # remove non-adults as I expect them not to be targets of the campaign, though they do seem to purchase the products sometimes.
    # remove also the clients with very high campaign revenue (removes 3 points)
    data = data[ data[ 'Age' ] > 17 ]
    data = data[ data[ 'Revenue_MF' ] < 150 ]
    data = data[ data[ 'Revenue_CC' ] < 250 ]
    data = data[ data[ 'Revenue_CL' ] < 50 ]
    data = data.drop( 'Client', axis = 1 )
    ToSetToZero = [ x for x in data.columns if 'Count' in x or 'ActBal' in x or 'Volume' in x or 'Transactions' in x ]
    data[ ToSetToZero ]=data[ ToSetToZero ].fillna(0)

    # split into the set i'll be working with and the set to provide predition for
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]
    data['color'] = data.apply( lambda row : set_color(row), axis=1) 
    data['weight'] = data.apply( lambda row : set_weight(row), axis=1) 

    col_names = data.columns.values.tolist()
    pairs = [(a, b) for idx, a in enumerate(col_names) for b in col_names[idx + 1:] if 'Sale' not in a and 'Sale' not in b 
                                                                                    and 'Revenue' not in a and 'Revenue' not in b 
                                                                                    and 'Sex' not in a and 'Sex' not in b 
                                                                                    and 'weight' not in a and 'weight' not in b 
                                                                                    and 'color' not in a and 'color' not in b  ]

    labels = [ 'MF', 'CC', 'CL', 'MFCC', 'MFCL', 'CLCC', 'all']
    colours = [ 'red', 'forestgreen', 'royalblue', 'orange', 'fuchsia', 'turquoise', 'gold' ]
    legend_elements = [ Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'red',         markersize = 7, label = 'MF'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'forestgreen', markersize = 7, label = 'CC'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'royalblue',   markersize = 7, label = 'CL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'orange',      markersize = 7, label = 'MFCC'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'fuchsia',     markersize = 7, label = 'MFCL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'turquoise',   markersize = 7, label = 'CCCL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'gold',        markersize = 7, label = 'all'),
                      ]

    for a,b in pairs:
        print( a, b )
        plt.figure(figsize = (24, 18))
        fig, ax = plt.subplots()
        ax.scatter( x = data[ a ],
                     y = data[ b ],
                     c = data[ 'color' ],
                     s = data[ 'weight' ],
                     edgecolors='none',
                     alpha = 0.5,
                     )
        plt.xlabel( a )
        plt.ylabel( b )
        ax.legend( handles=legend_elements, loc='upper center', bbox_to_anchor=( 1.025, 1.15))
#        plt.show()
        plt.savefig( 'plots/pdf/'+a+'_'+b+'.pdf' )
        plt.savefig( 'plots/png/'+a+'_'+b+'.png' )
        plt.close()
