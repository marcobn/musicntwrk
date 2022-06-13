def analysisNetwork(seq,ntx):

    ''' 
    •	generates the directional network of chord progressions in roman numeral form
    •	seq (int) – list of roman numerals for each chords extracted from the score
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        nameseq = pd.DataFrame([[seq[n]]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        pair = seq[n-1] == seq[n]
        if not pair:
            tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n])]],columns=['Source','Target'])
            dedges = dedges.append(tmp)


    if ntx:
        # evaluate average degree and modularity
        gbch = nx.from_pandas_edgelist(dedges,'Source','Target',create_using=nx.DiGraph())
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target')
        # modularity 
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)
