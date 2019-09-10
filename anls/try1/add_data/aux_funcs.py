
def nph(ar_x):
    """custom hist function because plt.hist is fucking unreliably"""
    a1,a2 = np.histogram(ar_x, bins='auto')
    # width = 0.7 * (a2[1] - a2[0])
    width = (a2[1] - a2[0])
    center = (a2[:-1] + a2[1:]) / 2
    plt.bar(center, a1, align='center', width=width)
    plt.show()

def nps(x,y, lbl):
    """custom scatter function"""
    plt.scatter(x,y)
    plt.show()

def npl(x, m = False):
    if m == True:
        [plt.plot(i) for i in x]
    else:
        plt.plot(x)
    plt.show()



# g = g_kld2
# ids = g_kld2_id
# filename = 'acst_spc5.pdf'
# eweit = kld_sim_int


def graph_pltr(g, ids, filename, eweit):
    """function for graph plotting, maybe put all the plotting parameters into function too?"""

    gt_lbls_plot = g.new_vertex_property('string')

    for v in g.vertices():
        x = ids[v]

        gt_lbls_plot[v] = x.replace(" ", "\n")

    size = g.degree_property_map('out')

    # size_scl=graph_tool.draw.prop_to_size(size, mi=1.5, ma=3, log=False, power=0.5)
    # size_scl=graph_tool.draw.prop_to_size(size, mi=3, ma=6, log=False, power=0.5)
    # size_scl=graph_tool.draw.prop_to_size(size, mi=4, ma=8, log=False, power=0.5)
    size_scl=graph_tool.draw.prop_to_size(size, mi=7, ma=25, log=False, power=0.5)
    # size_scl=graph_tool.draw.prop_to_size(size, mi=12, ma=50, log=False, power=0.5)

    size_scl2=graph_tool.draw.prop_to_size(size, mi=0.005, ma=0.1, log=False, power=1)
    # size_scl2=graph_tool.draw.prop_to_size(size, mi=0.005, ma=0.07, log=False, power=1)
    # size_scl2=graph_tool.draw.prop_to_size(size, mi=0.0025, ma=0.035, log=False, power=1)

    if type(eweit) == type(1.0):
        e_scl = eweit

    else:
        e_scl=graph_tool.draw.prop_to_size(eweit, mi=1, ma=6, log=False, power=0.5)

    gvd = graphviz_draw(g, size = (70,70),
                        # layout = 'sfdp',
                        # overlap = 'scalexy',
                        overlap = 'false',
                        vprops = {'xlabel':gt_lbls_plot, 'fontsize':size_scl, 'height':0.03,
                                  'shape':'point', 'fixedsize': True,
                                  'width':size_scl2, 'height':size_scl2, 'fillcolor':'black'},
                        eprops = {'arrowhead':'vee', 'color':'grey', 'weight':eweit,
                                  'penwidth':e_scl},
                        # returngv==True,
                        output = filename)
    gt_lbls_plot = 0
    


g = g_usrs_1md
eweit = g_usrs_1md_strng
filename = 'groups1.pdf'

def graph_pltr2(g, filename, eweit):
    """function for graph plotting, maybe put all the plotting parameters into function too?"""

    size = g.degree_property_map('out')

    # size_scl=graph_tool.draw.prop_to_size(size, mi=3, ma=6, log=False, power=0.5)
    # size_scl=graph_tool.draw.prop_to_size(size, mi=4, ma=8, log=False, power=0.5)
    # size_scl=graph_tool.draw.prop_to_size(size, mi=7, ma=25, log=False, power=0.5)
    size_scl=graph_tool.draw.prop_to_size(size, mi=12, ma=50, log=False, power=0.5)

    size_
    scl2=graph_tool.draw.prop_to_size(size, mi=0.025, ma=0.15, log=False, power=1)

    if type(eweit) == type(1):
        e_scl = eweit

    else:
        e_scl=graph_tool.draw.prop_to_size(eweit, mi=1, ma=6, log=False, power=0.5)

    gvd = graphviz_draw(g, size = (20,20),
                        # layout = 'sfdp',
                        # overlap = 'scalexy',
                        overlap = 'false',
                        vprops = {'height':0.03,'shape':'point', 'fixedsize': True,
                                  'width':size_scl2, 'height':size_scl2, 'fillcolor':'black'},
                        eprops = {'arrowhead':'vee', 'color':'grey', 'weight':eweit,
                                  'penwidth':eweit},
                        # returngv==True,
                        output = filename)
    gt_lbls_plot = 0
    

    
