import md5
import os

data_folder = '/data2_1e4'
dir_path = os.path.dirname(os.path.realpath(__file__))

def remove_duplicates(dir):
    unique = []
    for filename in os.listdir(dir):
    	filename_trimmed = shorten_filename(filename)
    	
    	#os.rename(dir + "/" + filename, dir + "/" + filename_trimmed)

def shorten_filename(name):

	split_file = name.split('_')
	if len(split_file) == 5:
		print(split_file)
		name = "_".join(split_file[:len(split_file) - 2])
		name = name + "_" + split_file[len(split_file) - 1]
	return name

if __name__ == "__main__":	
	remove_duplicates(dir_path + data_folder)
