# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:46:40 2018

@author: cyriac.azefack
"""

with open('default_template.py', 'r') as input_file :
    d = input_file.readlines()
    
    # You can put all the for loop here, candidate_size, etc...
    with open('output_candidate_size_etc.py', 'w') as output_file:
        d[15] = "papa16\n"
        d[16] = "papa17\n"
        d[17] = "papa19\n"
    
        output_file.writelines(d)
        output_file.close()
    input_file.close()


