import numpy as np

def str2row(row):
    numbers=np.empty([1,3])
    j_temp=0
    column_count=0
    for j in range(len(row)):
            if(row[j]==' '):
                temp_element=float(row[j_temp:j-1])
                j_temp=j
                numbers[0,column_count]=temp_element
                column_count=column_count+1
            if(j==len(row)-1):
                temp_element=float(row[j_temp+1:len(row)])
                numbers[0,column_count]=temp_element
                column_count=column_count+1
    return numbers
def h_matrix(h_path):
    with open(h_path) as f:
        lines = f.readlines()
    i_temp=18
    row_count=0
    homograph=np.zeros([3,3])
    for i in range(len(lines[0][20:len(lines[0])-2])):
        if(lines[0][i]==';'):
            temp_row=lines[0][i_temp+1:i]
            i_temp=i
            homograph[row_count,:]=str2row(temp_row)
            row_count=row_count+1
        if(i==len(lines[0][20:len(lines[0])-2])-1):
            if(lines[0][i]==' '):
                temp_row=lines[0][i_temp+1:len(lines[0])-1]
            else:
                temp_row=lines[0][i_temp+1:len(lines[0])]
            homograph[row_count,:]=str2row(temp_row)
    return homograph
