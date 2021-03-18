import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_exp import *


if __name__ == '__main__' :

    # load data, split into test and train set, look at general properties of data
    data = import_data( 'data/Task_Data_Scientist_Dataset.xlsx' )

    # remove non-adults as I expect them not to be targets of the campaign, though they do seem to purchase the products sometimes.
    # remove also the clients with very high campaign revenue (removes 3 points)
    i = 1
    ToSetToZero = [ x for x in data.columns if 'Count' in x or 'ActBal' in x or 'Volume' in x or 'Transactions' in x ]
    data[ ToSetToZero ]=data[ ToSetToZero ].fillna(0)
    data = data[ data[ 'Age' ] > 17 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'Revenue_MF' ] < 150 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'Revenue_CC' ] < 150 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'Revenue_CL' ] < 50 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeCred_CA' ] < 40000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeDeb_CA' ] < 30000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeDebCash_Card' ] < 5000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeDebCashless_Card' ] < 3000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeCred' ] < 40000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeDeb' ] < 32000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'VolumeDeb_PaymentOrder' ] < 20000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_CA' ] < 40000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_CC' ] < 1250 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_CL' ] < 15000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_MF' ] < 100000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_OVD' ] < 1000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data[ data[ 'ActBal_SA' ] < 50000 ]
    print( ++i, ' we have: ', data.shape[0])
    data = data.drop( 'Client', axis = 1 )
    ToSetToZero = [ x for x in data.columns if 'Count' in x or 'ActBal' in x or 'Volume' in x or 'Transactions' in x ]
    data[ ToSetToZero ]=data[ ToSetToZero ].fillna(0)

    # split into the set i'll be working with and the set to provide predition for
    data_predict = data[data['Sale_MF'].isna()]
    data = data[data['Sale_MF'].notna()]
    data['color'] = data.apply( lambda row : set_color(row), axis=1) 
    data['colorMF'] = data.apply( lambda row : set_colorMF(row), axis=1) 
    data['colorCC'] = data.apply( lambda row : set_colorCC(row), axis=1) 
    data['colorCL'] = data.apply( lambda row : set_colorCL(row), axis=1) 
    data['weight'] = data.apply( lambda row : set_weight(row), axis=1) 
    data['weightMF'] = data.apply( lambda row : set_weightMF(row), axis=1) 
    data['weightCC'] = data.apply( lambda row : set_weightCC(row), axis=1) 
    data['weightCL'] = data.apply( lambda row : set_weightCL(row), axis=1) 

    data = data.drop( 'Sex', axis = 1 )
    col_names = data.columns.values.tolist()
    pairs = [(a, b) for idx, a in enumerate(col_names) for b in col_names[idx + 1:] if 'Sale' not in a and 'Sale' not in b 
                                                                                    and 'Revenue' not in a and 'Revenue' not in b 
                                                                                    and 'Sex' not in a and 'Sex' not in b 
                                                                                    and 'weight' not in a and 'weight' not in b 
                                                                                    and 'color' not in a and 'color' not in b  ]

    legend_elements = [ Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'red',         markersize = 7, label = 'MF'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'forestgreen', markersize = 7, label = 'CC'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'royalblue',   markersize = 7, label = 'CL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'orange',      markersize = 7, label = 'MFCC'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'fuchsia',     markersize = 7, label = 'MFCL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'turquoise',   markersize = 7, label = 'CCCL'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'gold',        markersize = 7, label = 'all'),
                        Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'black',       markersize = 7, label = 'none')
                      ]
    legend_elementsMF = [ Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'forestgreen',         markersize = 7, label = 'MF'),
                          Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'royalblue',        markersize = 7, label = 'none')]
    legend_elementsCL = [ Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'forestgreen',         markersize = 7, label = 'CL'),
                          Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'royalblue',        markersize = 7, label = 'none')]
    legend_elementsCC = [ Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'forestgreen',         markersize = 7, label = 'CC'),
                          Line2D( [0], [0], marker = 'o', color = 'none', markeredgecolor = 'none', markerfacecolor= 'royalblue',        markersize = 7, label = 'none')]
                          
    for i in col_names:
        if 'weight' in i or 'color' in i or 'Sale' in i or 'Revenue' in i:
            continue
        print(i)
        plt.hist( [ data.loc[data[ "Sale_MF" ]==0][i],
                    data.loc[data[ "Sale_MF" ]==1][i],
                    ],
                    bins=20, stacked=False, color=[ 'royalblue', 'forestgreen' ] )
        plt.xlabel( i )
        plt.ylabel( 'Client count' )
        plt.legend( handles=legend_elementsMF, loc='upper center', bbox_to_anchor=( 0.9, 0.98))
        plt.yscale('linear')
        plt.savefig( 'plots/MF_hist/'+i+'.pdf' )
        plt.savefig( 'plots/MF_hist/'+i+'.png' )
        plt.yscale('log')
        plt.savefig( 'plots/MF_hist/'+i+'_log.pdf' )
        plt.savefig( 'plots/MF_hist/'+i+'_log.png' )
        plt.close()
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ns, bins, patches = ax1.hist( [ data.loc[data[ "Sale_MF" ]==0][i],
                                        data.loc[data[ "Sale_MF" ]==1][i],
                                        ],
                                        bins=20, stacked=False, alpha = 0.9, color=[ 'royalblue', 'forestgreen' ] )
        ax2.set_xlabel( i )
        ax2.set_ylabel( 'ratio' )
        ax1.set_ylabel( 'client count' )
        ns[0][ ns[0]==0 ]=1.
        to_plot =ns[1]/ns[0]
        ax2.plot(bins[:-1],
                 to_plot)
        plt.legend( handles=legend_elementsMF, loc='upper center', bbox_to_anchor=( 0.9, 2.1))
        plt.savefig( 'plots/MF_hist/'+i+'ratio.pdf' )
        plt.savefig( 'plots/MF_hist/'+i+'ratio.png' )
        ax1.set_yscale('log')
        plt.savefig( 'plots/MF_hist/'+i+'ratio_log.pdf' )
        plt.savefig( 'plots/MF_hist/'+i+'ratio_log.png' )
        plt.close()


    for i in col_names:
        if 'weight' in i or 'color' in i or 'Sale' in i or 'Revenue' in i:
            continue
        print(i)
        plt.hist( [ data.loc[data[ "Sale_CC" ]==0][i],
                    data.loc[data[ "Sale_CC" ]==1][i],
                    ],
                    bins=20, stacked=False, color=[ 'royalblue', 'forestgreen' ] )
        plt.xlabel( i )
        plt.ylabel( 'Client count' )
        plt.legend( handles=legend_elementsCC, loc='upper center', bbox_to_anchor=( 0.9, 0.98))
        plt.yscale('linear')
        plt.savefig( 'plots/CC_hist/'+i+'.pdf' )
        plt.savefig( 'plots/CC_hist/'+i+'.png' )
        plt.yscale('log')
        plt.savefig( 'plots/CC_hist/'+i+'_log.pdf' )
        plt.savefig( 'plots/CC_hist/'+i+'_log.png' )
        plt.close()
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ns, bins, patches = ax1.hist( [ data.loc[data[ "Sale_CC" ]==0][i],
                                        data.loc[data[ "Sale_CC" ]==1][i],
                                        ],
                                        bins=20, stacked=False, alpha = 0.9, color=[ 'royalblue', 'forestgreen' ] )
        ax2.set_xlabel( i )
        ax2.set_ylabel( 'ratio' )
        ax1.set_ylabel( 'client count' )
        ns[0][ ns[0]==0 ]=1.
        to_plot =ns[1]/ns[0]
        ax2.plot(bins[:-1],
                 to_plot)
        plt.legend( handles=legend_elementsCC, loc='upper center', bbox_to_anchor=( 0.9, 2.1))
        plt.savefig( 'plots/CC_hist/'+i+'ratio.pdf' )
        plt.savefig( 'plots/CC_hist/'+i+'ratio.png' )
        ax1.set_yscale('log')
        plt.savefig( 'plots/CC_hist/'+i+'ratio_log.pdf' )
        plt.savefig( 'plots/CC_hist/'+i+'ratio_log.png' )
        plt.close()


    for i in col_names:
        if 'weight' in i or 'color' in i or 'Sale' in i or 'Revenue' in i:
            continue
        print(i)
        plt.hist( [ data.loc[data[ "Sale_CL" ]==0][i],
                    data.loc[data[ "Sale_CL" ]==1][i],
                    ],
                    bins=20, stacked=False, color=[ 'royalblue', 'forestgreen' ] )
        plt.xlabel( i )
        plt.ylabel( 'Client count' )
        plt.legend( handles=legend_elementsCL, loc='upper center', bbox_to_anchor=( 0.9, 0.98))
        plt.yscale('linear')
        plt.savefig( 'plots/CL_hist/'+i+'.pdf' )
        plt.savefig( 'plots/CL_hist/'+i+'.png' )
        plt.yscale('log')
        plt.savefig( 'plots/CL_hist/'+i+'_log.pdf' )
        plt.savefig( 'plots/CL_hist/'+i+'_log.png' )
        plt.close()
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ns, bins, patches = ax1.hist( [ data.loc[data[ "Sale_CL" ]==0][i],
                                        data.loc[data[ "Sale_CL" ]==1][i],
                                        ],
                                        bins=20, stacked=False, alpha = 0.9, color=[ 'royalblue', 'forestgreen' ] )
        ax2.set_xlabel( i )
        ax2.set_ylabel( 'ratio' )
        ax1.set_ylabel( 'client count' )
        ns[0][ ns[0]==0 ]=1.
        to_plot =ns[1]/ns[0]
        ax2.plot(bins[:-1],
                 to_plot)
        plt.legend( handles=legend_elementsCL, loc='upper center', bbox_to_anchor=( 0.9, 2.1))
        plt.savefig( 'plots/CL_hist/'+i+'ratio.pdf' )
        plt.savefig( 'plots/CL_hist/'+i+'ratio.png' )
        ax1.set_yscale('log')
        plt.savefig( 'plots/CL_hist/'+i+'ratio_log.pdf' )
        plt.savefig( 'plots/CL_hist/'+i+'ratio_log.png' )
        plt.close()



    for a,b in pairs:
        print( a, b )
        plt.figure(figsize = (12, 9))
        plt.scatter( x = data[ a ],
                     y = data[ b ],
                     c = data[ 'color' ],
                     s = data[ 'weight' ],
                     edgecolors='none',
                     alpha = 0.5,
                     )
        plt.xlabel( a )
        plt.ylabel( b )
        plt.legend( handles=legend_elements, loc='upper center', bbox_to_anchor=( 1.025, 1.15))
#        plt.show()
        plt.savefig( 'plots/all_cats/'+a+'_'+b+'.pdf' )
        plt.savefig( 'plots/all_cats/'+a+'_'+b+'.png' )
        plt.close()
        break


    for a,b in pairs:
        print( a, b )
        plt.figure(figsize = (8, 6))
        plt.scatter( x = data[ a ],
                     y = data[ b ],
                     c = data[ 'colorMF' ],
                     s = data[ 'weightMF' ],
                     edgecolors='none',
                     alpha = 0.5,
                     )
        plt.xlabel( a )
        plt.ylabel( b )
        plt.legend( handles=legend_elementsMF, loc='upper center', bbox_to_anchor=( 1.025, 1.15))
        plt.savefig( 'plots/MF/'+a+'_'+b+'.pdf' )
        plt.savefig( 'plots/MF/'+a+'_'+b+'.png' )
        plt.close()


    for a,b in pairs:
        print( a, b )
        plt.figure(figsize = (8, 6))
        plt.scatter( x = data[ a ],
                     y = data[ b ],
                     c = data[ 'colorCC' ],
                     s = data[ 'weightCC' ],
                     edgecolors='none',
                     alpha = 0.5,
                     )
        plt.xlabel( a )
        plt.ylabel( b )
        plt.legend( handles=legend_elementsCC, loc='upper center', bbox_to_anchor=( 1.025, 1.15))
        plt.savefig( 'plots/CC/'+a+'_'+b+'.pdf' )
        plt.savefig( 'plots/CC/'+a+'_'+b+'.png' )
        plt.close()


    for a,b in pairs:
        print( a, b )
        plt.figure(figsize = (8, 6))
        plt.scatter( x = data[ a ],
                     y = data[ b ],
                     c = data[ 'colorCL' ],
                     s = data[ 'weightCL' ],
                     edgecolors='none',
                     alpha = 0.5,
                     )
        plt.xlabel( a )
        plt.ylabel( b )
        plt.legend( handles=legend_elementsCL, loc='upper center', bbox_to_anchor=( 1.025, 1.15))
#        plt.show()
        plt.savefig( 'plots/CL/'+a+'_'+b+'.pdf' )
        plt.savefig( 'plots/CL/'+a+'_'+b+'.png' )
        plt.close()
