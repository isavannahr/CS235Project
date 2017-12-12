def main_association():
  #open mass shootings data set and create a dictionary
  filename_ms = "Datasets/mass_shootings_2014-2017.csv";
  file_ms = open(filename_ms, "r");
  array_ms={}
  count=1;
  for line in file_ms:
    if count>1:
      line_arr=line.split(',')
      index=line_arr[1]+'-'+line_arr[0]
      #print index
      array_ms[index]=line_arr[2]
    count+=1

#  for i,v in array_ms.items():
#    print i+'-'+v

  filename_gl = "Datasets/num_gun_laws_by_state_per_year_2014-2017.csv";
  file_gl = open(filename_gl, "r");
  header_gl=['ammunition_regulations','assault_weapons_regulations','background_checks','buyer_regulations','child_access_prevention','concealed_carry_permitting','dealer_regulations','domestic_violence','gun_trafficking','immunity','no_stand_your_ground','possession_regulations','preemption','prohibitions_against_high_risk_gun_owners','lawtotal']
  array_gl={}
  count=1;
  array_sums={}
  for line in file_gl:
    if count>1:
      line_arr=line.split(',')
      index=line_arr[0]+'-'+line_arr[1]
      temp_map={}
      array_sums[index]=0
      count2=2;
      for h in header_gl:
        temp_map[h]=line_arr[count2]
        #array_sums[index]+=temp_map[h]
        count2+=1
      temp_map['mass_shootings']=array_ms[index]
      array_gl[index]=temp_map
      #array_sums[index]+=temp_map['mass_shootings']
     
    count+=1
  #i= year-state
  #j= gun law type
  array_support={}
  for i,v in array_gl.items():
     array_support[i]={}
     for j,x in v.items():
       if j!='lawtotal' and j!='mass_shootings':
         y=float(array_gl[i]['mass_shootings'])
         if x>y:
           t=y
         else:
           t=x
         if y>0:
           array_support[i][j]=(float(t)/y)
         elif y<1:
           array_support[i][j]='N/A'
         elif x<1:
           array_support[i][j]=0
         else:
           array_support[i][j]=(float(t)/float(y))
       

  f = open('output_association.csv','w')
  f2 = open('more_laws_than_shootings.csv','w')
  f3 = open('more_shootings_than_laws.csv','w')
  f.write('year,state,gun_law_type,num_laws,num_mass_shootings,support_by_law_type\n');
  f2.write('year,state,num_laws,num_mass_shootings,overall_support\n');
  f3.write('year,state,num_laws,num_mass_shootings,overall_support\n');
  for i,v in array_support.items():
    for j,x in v.items():
      line_arr=i.split('-')
      num_laws=array_gl[i][j]
      totallaws=int(array_gl[i]['lawtotal'])
      num_ms=int(array_gl[i]['mass_shootings'])
      if float(num_ms)>0:
        if totallaws>num_ms:
          t=num_ms
        else:
          t=totallaws
        overall_support=float(t)/float(num_ms)
      else:
        overall_support='N/A'
      f.write(str(line_arr[0])+','+str(line_arr[1])+','+str(j)+','+str(num_laws)+','+str(num_ms)+','+str(x)+'\n')
      if totallaws>num_ms:
        f2.write(str(line_arr[0])+','+str(line_arr[1])+','+str(totallaws)+','+str(num_ms)+','+str(overall_support)+'\n')
      if num_ms>=totallaws:
        f3.write(str(line_arr[0])+','+str(line_arr[1])+','+str(totallaws)+','+str(num_ms)+','+str(overall_support)+'\n')
      
