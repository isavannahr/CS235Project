import sys;
#include files that include our individual functions here
#from association.py import function main_association
from association import main_association

ans=True
while ans:
    print ("""
    1.Association
    2.Clustering
    3.Sequential Patterns
    4.Exit/Quit
    """)
    ans=raw_input("Choose a Data Mining Technique: ") 
    if ans=="1": 
        #call to Claudia's function for Association goes here
         main_association();
    elif ans=="2":
        #call to function
         print("function2");
    elif ans=="3":
       #call to function
        print("function3");
    elif ans=="4":
       #call to function
       sys.exit();
    elif ans !="":
      print("\n Please select an option from 1-4") 
