# Makefile to generate the executable files
# Coded and annotated by Yuxuan SONG
# Last modified time: Jue 3, 2024
all:
	nvcc -g spmmCSRBatched.cu -lcusparse -o spmmCSRBatched
	nvcc -g gemmBatched.cu -lcublas -o gemmBatched
	nvcc -g spmmBRCVBatched_v1.cu -o spmmBRCVBatched
clean:
	rm spmmCSRBatched
	rm gemmBatched
	rm spmmBRCVBatched