import requests
import numpy as np
ConceptNetAdd = 'http://api.conceptnet.io/'

def CNetNode(word):
    obj = requests.get(ConceptNetAdd+"c/en/"+word).json()
    return obj

def CNetQuery(dict_in):
    #dict_in = {'start':node,'end': node, 'rel': relation,'node': 'start or 'end'}
    qin = ConceptNetAdd + "query?"
    for key, value in dict_in.items():
        if key in 'rel':
            qin = qin + 'rel=/r/'+value + '&'
        else:
            qin = qin + key + '=/c/en/'+value+'&'
    qin = qin[:-1]

    #print(qin)
    obj = requests.get(qin).json()
    return obj

def CNetGetScore(node1, node2, rel): #rel 'in': 'node1 is in node2', 'usedfor': node1 is usedfor node2
    if rel in 'in':
        obj = CNetQuery({'start':node1, 'end':node2, 'rel':'AtLocation'})

    elif rel in 'usedfor':
        obj = CNetQuery({'start': node1, 'end': node2, 'rel': 'UsedFor'})

    if len(obj['edges']) > 0 :
        score = obj['edges'][0]['weight']
    else:
        score = 0

    return score

def CNetScoreObjRoom(objlist, roomlist):
    score_mat = np.ndarray((len(objlist), len(roomlist)))
    for ii in range(0, len(objlist)):
        for jj in range(0, len(roomlist)):
            score_mat[ii, jj] = CNetGetScore(objlist[ii], roomlist[jj], 'in')

    return score_mat

if __name__== '__main__':
    score = CNetGetScore('sofa', 'bedroom', 'in')
    score_mat = CNetScoreObjRoom(['fridge', 'chair', 'table', 'sofa', 'coffee_table'], ['bedroom', 'living_room', 'garage', 'kitchen'])

    print(score_mat)

