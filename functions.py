import numpy as np

def output_data(Nline_new, Ncol_new, T_fit, c_fit, s, M_fit):
  
  ref_file = open("SpecificHeat.txt","w")
  for i in range(Nline_new):

      if i == 0:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write("T")
              else:
                  ref_file.write(f"\t{(j-1)}T")
      else:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write(str(T_fit[i-1]))
              else:
                  ref_file.write(f"\t{c_fit[i-1,j-1]}")

      ref_file.write('\n')
  ref_file.close()

  ref_file = open("TotalEntropy.txt","w")
  for i in range(Nline_new):

      if i == 0:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write("T")
              else:
                  ref_file.write(f"\t{(j-1)}T")
      else:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write(str(T_fit[i-1]))
              else:
                  ref_file.write(f"\t{s[i-1,j-1]}")

      ref_file.write('\n')
  ref_file.close()

  ref_file = open("Magnetization.txt","w")
  for i in range(Nline_new):

      if i == 0:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write("T")
              else:
                  ref_file.write(f"\t{(j-1)}T")
      else:
          for j in range(Ncol_new+1):
              if j == 0:
                  ref_file.write(str(T_fit[i-1]))
              else:
                  ref_file.write(f"\t{M_fit[i-1,j-1]}")

      ref_file.write('\n')
  ref_file.close()
