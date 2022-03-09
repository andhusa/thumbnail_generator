from cgitb import text
from distutils import command
from fileinput import filename
from logging import root
from sqlite3 import Row
from sre_parse import State
from tokenize import String
from turtle import title, width
from cv2 import blur

from numpy import column_stack, pad, tri
from sshtunnel import SSHTunnelForwarder
import subprocess
import os
import paramiko
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
from scp import SCPClient
from PIL import ImageTk, Image
import time

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def generate():
	filename = fileName.get().split('/')[-1]
	name, ext = os.path.splitext(filename)
	outputname = name + '_thumbnail.jpg'

	
	print("Number frames to extract: %s\nFace detection model: %s\nFilename: %s" % (fe.get(), faceVariable.get(), fileName.get()))
	ssh = createSSHClient('dnat.simula.no', 60441, 'andrehus', 'admin')
	scp = SCPClient(ssh.get_transport())
	scp.put(fileName.get(), 'egne_prosjekter/videoAndOutput/')
	vidName = fileName.get().split("/")[-1]
	csStr = "-css " + cs.get()
	ceStr = "-ces " + ce.get()
	closeUpThrStr = "-cuthr " + str(float(cuthr.get())/100)
	nfeStr = '-nfe ' + fe.get()
	runIQAStr = "-xi" if not runIQA.get() else "-brthr " + brisque.get()
	runLogoDetectionStr = "-xl" if not runLogoDetection.get() else ""
	runFaceDetectionStr = "-xf" if not runFaceDetection.get() else "-" + faceVariable.get()
	nbytes = 4096
	hostname = 'dnat.simula.no'
	port = 60441
	username = 'andrehus' 
	password = 'admin'
	command = '. activate_conda_environment.sh\n cd egne_prosjekter/thumbnail_generator/\n python create_thumbnail.py %s %s %s %s %s %s %s %s\n rm ../videoAndOutput/%s' % ("../videoAndOutput/" + filename, runFaceDetectionStr, closeUpThrStr, nfeStr, runLogoDetectionStr, runIQAStr, csStr, ceStr, vidName)
	client = paramiko.Transport((hostname, port))
	client.connect(username=username, password=password)
	stdout_data = []
	stderr_data = []
	session = client.open_channel(kind='session')
	session.exec_command(command)
	while True:
		if session.recv_ready():
			stdout_data.append(session.recv(nbytes))
		if session.recv_stderr_ready():
			stderr_data.append(session.recv_stderr(nbytes))
		if session.exit_status_ready():
			break

	print('exit status: ', session.recv_exit_status())
	print(str(stdout_data))
	print(str(stderr_data))

	session.close()
	client.close()
	scp.get('egne_prosjekter/thumbnail_generator/thumbnail_output/' + outputname)
	
	img = Image.open(outputname)
	print(img)
	img = img.resize((width,height), Image.ANTIALIAS)
	print(img)
	photoImg = ImageTk.PhotoImage(img)
	print(photoImg)
	master.img = photoImg
	canvas.create_image(20,20, anchor=NW, image=photoImg)
	

def open_file():
	video_file = askopenfilename()
	video_file_text.set(video_file)
	fileName.delete(0, 'end')
	fileName.insert(0, video_file)
def display_face_det_models():
	if runFaceDetection.get():
		faceDetDropDown.config(state='normal')
	else:
		faceDetDropDown.config(state='disabled')
	
def display_brisque_thr():
	if runIQA.get():
		brisque.config(state='normal')
		iqaDropDown.config(state='normal')
	else:
		brisque.config(state='disabled')
		iqaDropDown.config(state='disabled')

def display_blur_thr():
	if runBlurDetection.get():
		blurThr.config(state='normal')
		blurDetDropDown.config(state='normal')
	else:
		blurThr.config(state='disabled')
		blurDetDropDown.config(state='disabled')
def display_logo_det_models():
	if runLogoDetection.get():
		logoDetDropDown.config(state='normal')
		ldthr.config(state='normal')
	else:
		logoDetDropDown.config(state='disabled')
		ldthr.config(state='disabled')

def disable_other_entries_downsampling():
	if down_sampling_var.get() == 1:
		fe.config(state='normal')
		dr.config(state='disabled')
		fps.config(state='disabled')
	if down_sampling_var.get() == 2:
		fe.config(state='disabled')
		dr.config(state='normal')
		fps.config(state='disabled')
	if down_sampling_var.get() == 3:
		fe.config(state='disabled')
		dr.config(state='disabled')
		fps.config(state='normal')
def display_close_up():
	if runCloseUpDetection.get():
		cuthr.config(state='normal')
		closeUpDropDown.config(state='normal')
	else:
		cuthr.config(state='disabled')
		closeUpDropDown.config(state='disabled')

master = tk.Tk()
master.winfo_toplevel().title("HOST-ATS Graphical User Interface")
faceVariable = StringVar(master)
faceVariable.set("dlib") # default value
logoVariable = StringVar(master)
logoVariable.set('Surma')
blurVariable = StringVar(master)
blurVariable.set("SVD")
iqaVar = StringVar(master)
iqaVar.set("Ocampo")
closeUpVar = StringVar(master)
closeUpVar.set("Surma")
runIQA = BooleanVar(value=True)
runLogoDetection = BooleanVar(value=True)
runFaceDetection = BooleanVar(value=True)
runBlurDetection = BooleanVar(value=True)
runCloseUpDetection = BooleanVar(value=True)
video_file_text = StringVar(value="No file selected					")
generate_button_state = StringVar(value='disabled')
pre_processing = LabelFrame(master, text="Step 1. Pre-processing", font=('Arial', 20), padx=10, pady=10)
pre_processing.grid( padx=10, pady=10)
trimming = LabelFrame(pre_processing, text="1a. Trimming", padx=10, pady=10)
trimming.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(trimming, text= "Annotation mark").grid(sticky=tk.W)
tk.Label(trimming, text= "Drop frames X seconds before the annotation").grid(sticky=tk.W)
tk.Label(trimming, text= "Drop frames Y seconds after the annotation").grid(sticky=tk.W)
tk.Label(trimming, text="Seconds to cut in start of video").grid(sticky=tk.W)
tk.Label(trimming, text="Seconds to cut in end of video").grid(sticky=tk.W)
down_sampling = LabelFrame(pre_processing, text="1b. Down-sampling", padx=10, pady=10)
down_sampling.grid(padx=10, pady=10, sticky=tk.W)
down_sampling_var = IntVar()
down_sampling_var.set(1)
tk.Radiobutton(down_sampling, text="Number of Frames to extract", variable=down_sampling_var, value=1, command=disable_other_entries_downsampling).grid(sticky=tk.W)
tk.Radiobutton(down_sampling, text="Downsampling ratio: value between [0,1]", variable=down_sampling_var, value=2, command=disable_other_entries_downsampling).grid(sticky=tk.W)
tk.Radiobutton(down_sampling, text="Frame per second", variable=down_sampling_var, value=3, command=disable_other_entries_downsampling).grid(sticky=tk.W)
down_scaling = LabelFrame(pre_processing, text="1c. Down-scaling", padx=10, pady=10)
down_scaling.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(down_scaling, text="For internal processing (%)").grid(sticky=tk.W)
tk.Label(down_scaling, text="For output image (%)").grid(sticky=tk.W)

content_analysis = LabelFrame(master, text="Step 2. Content Analysis", font=('Arial', 20), padx=10, pady=10)
content_analysis.grid(row=0, column=3, padx=10, pady=10, sticky=tk.N)
logo_detection = LabelFrame(content_analysis, text="2a. Logo Detection", padx=10, pady=10)
logo_detection.grid(padx=10, pady=10, sticky=tk.W)
logo = tk.Checkbutton(logo_detection, command=display_logo_det_models, text="Run Logo Detection", variable=runLogoDetection)
tk.Label(logo_detection, text="Logo Detection Model").grid(row=1, sticky=tk.W)
tk.Label(logo_detection, text="Logo detection threshold (%)").grid(row=2, sticky=tk.W)
close_up_shot_detection = LabelFrame(content_analysis, text="2b. Close-up Shot Detection", padx=10, pady=10)
close_up_shot_detection.grid(padx=10, pady=10)
runClose = tk.Checkbutton(close_up_shot_detection, command=display_close_up,text="Run Close-up Shot Detection", variable=runCloseUpDetection)
tk.Label(close_up_shot_detection, text="Close-up shot detection model").grid(row=1, sticky=tk.W)
tk.Label(close_up_shot_detection, text="Close-up detection threshold (%)").grid(row=2, sticky=tk.W)
face_detection = LabelFrame(content_analysis, text="2c. Face Detection", padx=10, pady=10)
face_detection.grid(padx=10, pady=10, sticky=tk.W)
runFace = tk.Checkbutton(face_detection, command=display_face_det_models, text="Run Face Detection", variable=runFaceDetection)
tk.Label(face_detection, text="Face Detection Model").grid(row=1, sticky=tk.W)

image_quality_analysis = LabelFrame(master, text="Step 3. Image Quality Analysis", font=('Arial', 20), padx=10, pady=10)
image_quality_analysis.grid(row=0, column=5, padx=10, pady=10, sticky=tk.N)
image_quality_prediction = LabelFrame(image_quality_analysis, text="3a. Image Quality Prediction", padx=10, pady=10)
image_quality_prediction.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(image_quality_prediction, text="Image quality prediction model").grid(row=1, sticky=tk.W)

tk.Label(image_quality_prediction, text="BRISQUE threshold value").grid(row=2, sticky=tk.W)
blur_detection = LabelFrame(image_quality_analysis, text="3b. Blur Detection", padx=10, pady=10)
blur_detection.grid(padx=10, pady=10, sticky=tk.W)
tk.Label(blur_detection, text="Blur Detection Model").grid(row=1, sticky=tk.W)
tk.Label(blur_detection, text="Blur score threshold").grid(row=2, sticky=tk.W)

file_processing = LabelFrame(master, padx=10, pady=10)
file_processing.grid(row=1, column=5, padx=10, pady=10, sticky=tk.N)
video_file_text_label = tk.Label(file_processing, textvariable=video_file_text, font=('Arial', 8)).grid(sticky=tk.W, column=1)
#generate_button = tk.Button(file_processing, text='Generate', command=generate).grid(sticky=tk.W)
generate_button = tk.Button(file_processing, text='Generate').grid(sticky=tk.W)


am = tk.Entry(trimming, width=5)
ba = tk.Entry(trimming, width=5)
aa = tk.Entry(trimming, width=5)
cs = tk.Entry(trimming, width=5)
ce = tk.Entry(trimming, width=5)
fe = tk.Entry(down_sampling, width=5)
dr = tk.Entry(down_sampling, width=5, state='disabled')
fps = tk.Entry(down_sampling, width=5, state='disabled')
ip = tk.Entry(down_scaling, width=5)
oi = tk.Entry(down_scaling, width=5)

ldthr = tk.Entry(logo_detection, width=5)
cuthr = tk.Entry(close_up_shot_detection, width=5)

faceDetDropDown = tk.OptionMenu(face_detection, faceVariable, "haar", "dlib", "mtcnn", "dnn")
logoDetDropDown = tk.OptionMenu(logo_detection, logoVariable, "Surma", "Ocampo")
blurDetDropDown = tk.OptionMenu(blur_detection, blurVariable, "SVD", "Laplacian")
iqaDropDown = tk.OptionMenu(image_quality_prediction, iqaVar, "Ocampo")
closeUpDropDown = tk.OptionMenu(close_up_shot_detection, closeUpVar, "Surma")
fileName = tk.Entry(file_processing)
iqa = tk.Checkbutton(image_quality_prediction, command=display_brisque_thr, text="Run Image Quality Prediction", variable=runIQA)
brisque = tk.Entry(image_quality_prediction, width=5)
runBlur = tk.Checkbutton(blur_detection, command=display_blur_thr, text="Run Blur Detection", variable=runBlurDetection)
blurThr = tk.Entry(blur_detection, width=5)
fe.insert(10, "50")
cs.insert(10, "0")
ce.insert(10, "0")
ip.insert(10, "50")
oi.insert(10, "100")
ldthr.insert(10, "10")
cuthr.insert(10, "75")
brisque.insert(10, "35")
blurThr.insert(10, "0.6")

am.grid(row=0, column=1)
ba.grid(row=1, column=1)
aa.grid(row=2, column=1)
cs.grid(row=3, column=1)
ce.grid(row=4, column=1)
fe.grid(row=0, column=1)
dr.grid(row=1, column=1)
fps.grid(row=2, column=1)
ip.grid(row=0, column=1)
oi.grid(row=1, column=1)

logo.grid(row=0, column=0, sticky=tk.W)
logoDetDropDown.grid(row=1, column=1)
ldthr.grid(row=2, column=1)

runClose.grid(row=0, column=0)
closeUpDropDown.grid(row=1, column=1)
cuthr.grid(row=2, column=1)

runFace.grid(row=0, column=0)
faceDetDropDown.grid(row=1, column=1)

iqa.grid(row=0, column=0)
iqaDropDown.grid(row=1, column=1)
brisque.grid(row=2, column=1)

runBlur.grid(row=0, column=0)
blurDetDropDown.grid(row=1,column=1)
blurThr.grid(row=2,column=1)




tk.Button(file_processing,
		text='Select File',
		command=open_file).grid(row=0, sticky=tk.W)

ratio = 1.7777778
height = 140
width = int(height * ratio)
canvas = Canvas(file_processing, width=width, height=height)
canvas.grid(sticky=tk.W)
master.mainloop()

tk.mainloop()

