from math import sqrt, atan

def Xs_formation(final_vector):
  Xs = []
  for joint in final_vector:
    Xs.append(joint[0][0])
    Xs.append(joint[0][1])
  return Xs

def new_get_H(Xs):
    if Xs[17]!=None:
        H = Xs[17] - Xs[3]
    else:
        H = Xs[23] - Xs[3]
    return H

def new_get_X(Xs,H,diction):
  sum = 0
  count = 0
  for value in Xs:
    if value !=None:
      sum+=value
      count+=1
  mean  = sum/count
  for i,value in enumerate(Xs):
    if i%2 == 0:
      if value!=None:
        diction['X_'+str(i//2)] = value-mean/H
      else:
        diction['X_'+str(i//2)] = None
    else:
      if value!=None:
        diction['Y_'+str(i//2)] = value-mean/H
      else:
        diction['Y_'+str(i//2)] = None
  return diction

def Djoints(vector,diction):
  i_is_None = False
  for i in range(0,len(vector)-3,2):
    if [vector[i],vector[i+1]]==[None,None]:
      i_is_None = True
    for j in range(i+2,len(vector),2):
      if (i_is_None==True) or ([vector[j],vector[j+1]]==[None,None]):
        diction['Dist_'+str(i//2)+'_'+str(j//2)] = None
        diction['Angle_'+str(i//2)+'_'+str(j//2)] = None
        continue
      dx = vector[j] - vector[i]
      dy = vector[j+1]- vector[i+1]
      distance = sqrt(pow(dx,2)+pow(dy,2))
      diction['Dist_'+str(i//2)+'_'+str(j//2)] = distance
      if dx==0:
        angle = 90 if dy>0 else -90
      else:
        angle = atan(dy/dx)
      diction['Angle_'+str(i//2)+'_'+str(j//2)] = angle
    i_is_None = False
  return diction
  
def dynamic_Djoints(vector,diction):
  i_is_None = False
  resultant = {}
  for i in range(0,len(vector)-3,2):
    if [vector[i],vector[i+1]]==[None,None]:
      i_is_None = True
    for j in range(i+2,len(vector),2):
      if (i_is_None==True) or ([vector[j],vector[j+1]]==[None,None]):
        resultant['Dist_'+str(i//2)+'_'+str(j//2)] = None
        resultant['Angle_'+str(i//2)+'_'+str(j//2)] = None
        continue
      dx = vector[j] - vector[i]
      dy = vector[j+1]- vector[i+1]
      distance = sqrt(pow(dx,2)+pow(dy,2))
      resultant['Dist_'+str(i//2)+'_'+str(j//2)] = distance
      if dx==0:
        angle = 90 if dy>0 else -90
      else:
        angle = atan(dy/dx)
      resultant['Angle_'+str(i//2)+'_'+str(j//2)] = angle
    i_is_None = False
  return resultant
