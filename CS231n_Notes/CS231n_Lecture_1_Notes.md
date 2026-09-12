# CS231n Lecture 1: Introduction

## Slide 1

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
Lecture 1: 
Introduction
4/3/2018 1

## Slide 2

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungWelcome to CS231n
2
Top	row,	left	to	right:
Image by	Roger	H	Goun is	licensed	under	 CC	BY	 2.0
Image is	CC0	1.0 public	 domain
Image is	CC0	1.0 public	domain
Image is	CC0	1.0 public	domain
Middle	row,	left	to	right
Image by	BGPHP	Conference is	licensed	under	 CC	BY	 2.0;	changes	made
Image is	CC0	1.0 public	domain
Image by	NASA is	licensed	under	 CC	BY	 2.0
Image is	CC0	1.0 public	domainBottom	row,	left	to	right
Image is	CC0	1.0 public	domain
Image by	Derek	Keats is	licensed	under	 CC	BY	2.0 ;	changes	made
Image is	public	 domain
Image	 is	licensed	under	 CC-BY	2.0 ;	changes	made
4/3/2018

## Slide 3

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
Computer 
Vision
Neuroscience
Machine learning
Speech, NLP
Information retrieval
MathematicsComputer
ScienceBiology
EngineeringPhysics
Robotics
Cognitive 
sciencesPsychology
graphics, algorithms, 
theory,…
Image
processing
3
systems, 
architecture, …
optics
4/3/2018

## Slide 4

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
Computer 
Vision
Neuroscience
Machine learning
Speech, NLP
Information retrieval
MathematicsComputer
ScienceBiology
EngineeringPhysics
Robotics
Cognitive 
sciencesPsychology
graphics, algorithms, 
theory,…
Image
processing
4
systems, 
architecture, …
optics
4/3/2018

## Slide 5

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungRelated Courses @ Stanford
•CS131: Computer Vision: Foundations and Applications
–Fall 2017, Juan Carlos Niebles and Ranjay Krishna
–Undergraduate introductory class
•CS231a: Computer Vision, from 3D Reconstruction to Recognition
–Winter 2018, Professor Silvio Savarese
–Core computer vision class for seniors, masters, and PhDs
–Image processing, cameras, 3D reconstruction, segmentation, object 
recognition, scene understanding; not just deep learning
•CS 224n: Natural Language Processing with Deep Learning
–Winter 2018, Richard Socher
•CS 230: Deep Learning
–Spring 2018, Prof. Andrew Ng and Kian Katanforoosh
•CS231n: Convolutional Neural Networks for Visual Recognition
–This course, Prof. Fei -Fei Li & Justin Johnson & Serena Yeung
–Focusing on applications of deep learning to computer vision
5 4/3/2018

## Slide 6

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungToday’s agenda
•A brief history of computer vision
•CS231n overview
6 4/3/2018

## Slide 7

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 7543 million years, B.C.
This	image is	licensed	under	 CC-BY	3.0This	image is	licensed	under	 CC-BY	2.5
This	image is	licensed	under	 CC-BY	2.5Evolution’s Big Bang
4/3/2018

## Slide 8

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungCamera Obscura
8
Leonardo da Vinci,
16thCentury AD
This	work	 is	in	the public	 domain
This	work	 is	in	the	public	domainGemma Frisius , 1545
This	 work	 is	in	the	public	domainEncyclopedie , 18thCentury
4/3/2018

## Slide 9

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 9
Stimulus
Electrical 
signal from 
brain
Stimulus Response
Catimage by	CNX	 OpenStax is	licensed	
under	 CC	BY	4.0 ;	changes	madeSimple cells : 
Response to light 
orientation
Complex cells:
Response to light 
orientation and movement
Hypercomplex cells : 
response to movement 
with an end pointHubel & Wiesel, 1959
No response Response 
(end point)
4/3/2018

## Slide 10

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 10Block world
Larry Roberts, 1963
(a) Original picture (b) Differentiated picture (c) Feature points selected
4/3/2018

## Slide 11

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 11 4/3/2018

## Slide 12

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 12
David Marr, 1970s
4/3/2018

## Slide 13

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 13
This	image is	CC0	1.0 public	domain
This	image is	CC0	1.0 public	domainInput image Edge image2 ½-D sketch 3-D model
Input
Image
Perceived
intensitiesPrimal
Sketch
Zero crossings,
blobs, edges, 
bars, ends, 
virtual lines,
groups, curves 
boundaries2 ½-D
Sketch
Local surface 
orientation 
and 
discontinuities 
in depth and 
in surface 
orientation3-D Model
Representation
3-D models 
hierarchically 
organized in 
terms of 
surface and 
volumetric 
primitives
Stages of Visual Representation, David Marr, 1970s
4/3/2018

## Slide 14

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 14•Generalized Cylinder •Pictorial Structure
Brooks & Binford , 1979 Fischler and Elschlager , 1973 
b
b
4/3/2018

## Slide 15

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 15David Lowe, 1987Image	 is	CC0	1.0	 public	domain
4/3/2018

## Slide 16

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 16Normalized Cut (Shi & Malik, 1997)
Image is	CC	BY	3.0
Image is	public	domainImage	 is	CC-BY	2.0 ;	
changes	made
4/3/2018

## Slide 17

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 17Face Detection, Viola & Jones, 
2001
Image is	public	
domain
4/3/2018

## Slide 18

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 18“SIFT” & Object Recognition, David Lowe, 1999
Image is	public	domainImage is	public	domain
4/3/2018

## Slide 19

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 19Spatial Pyramid Matching, Lazebnik , Schmid & Ponce, 2006
Level	0 Level	1
Image is	CC0	1.0 public	domain
4/3/2018

## Slide 20

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 20Histogram of Gradients ( HoG )
Dalal & Triggs , 2005Deformable Part Model
Felzenswalb , McAllester , Ramanan , 2009
orientationfrequency
Image is	CC0	1.0 public	domain
4/3/2018

## Slide 21

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 21PASCAL Visual Object Challenge 
(20 object categories)
[Everingham et al. 2006 -2012]
Airplane
Image is	CC0	1.0 public	domain
Train
Person
Image is	CC0	1.0 public	 domain
Image is	CC0	1.0 public	 domain
4/3/2018

## Slide 22

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 22
22K categories	and	 14M images
www.image
 -
net.org
Deng, Dong, Socher , Li, Li, & Fei -Fei, 2009 •Animals
•Bird
•Fish
•Mammal
•Invertebrate•Plants
•Tree
•Flower
•Food
•Materials•Structures
•Artifact
•Tools
•Appliances
•Structures•Person
•Scenes
•Indoor
•Geological	Formations
•Sport	Activities
4/3/2018

## Slide 23

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 23
Output:
Scale
T-shirt
Steel drum
Drumstick
Mud turtleSteel	drum
✔ ✗Output:
Scale
T-shirt
Giant panda
Drumstick
Mud turtle
4/3/2018Russakovsky et al. IJCV 2015The Image Classification Challenge:
1,000 object classes
1,431,167 images

## Slide 24

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
 24
Steel	drumThe Image Classification Challenge:
1,000 object classes
1,431,167 images
4/3/2018Russakovsky et al. IJCV 2015

## Slide 25

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungToday’s agenda
•A brief history of computer vision
•CS231n overview
25 4/3/2018

## Slide 26

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungCS231n focuses on one of the most fundamental 
problems of visual recognition –
image classification
26 4/3/2018

## Slide 27

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 27
Image by	Kippelboy is	licensed	under	 CC	BY -SA	3.0
 Image by	Christina	C.	is	licensed	under	 CC	BY -SA	4.0
Image by	US	Army	 is	licensed	under	 CC	BY	2.0 Image is	CC0	1.0 public	domain
4/3/2018

## Slide 28

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungThere are many visual recognition problems that 
are related to image classification, such as 
object detection, image captioning
28 4/3/2018

## Slide 29

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 29•Object detection
•Action classification
•Image captioning
•…
This	image is	licensed	under	 CC	BY -NC-SA	2.0 ;	changes	made
Person
Hammer
This	image is	licensed	under	 CC	BY -SA	2.0 ;	changes	made
Person BikePerson	on	Bike
This	image is	licensed	under	 CC	BY -SA	3.0 ;	changes	made
4/3/2018

## Slide 30

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungConvolutional Neural Networks (CNN) have 
become an important tool for object recognition
30 4/3/2018

## Slide 31

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungGoogLeNet VGG MSRA
 SuperVision
[Krizhevsky NIPS 2012]
Year 2012 Year 2014
 Year 2010
NEC-UIUC
[Lin CVPR 2011]
[Szegedy arxiv 2014] [Simonyan arxiv 2014]
31
Year 2015
Dense	descriptor	grid:	
HOG,	LBP
Coding:	local	coordinate,	
super -vector
Pooling,	SPM
Linear	SVM
Lion	image by	Swissfrog is	
licensed	 under	 CC	BY	3.0Image
conv -64
conv -64
maxpool
conv -128
conv -128
maxpool
conv -256
conv -256
maxpool
conv -512
conv -512
maxpool
fc-4096
fc-4096
fc-1000
softmaxconv -512
conv -512
maxpool
Pooling
Convolutio
n
Softmax
Other
[He ICCV 2015] Figure	copyright	Alex	 Krizhevsky ,	Ilya	
Sutskever ,	and	Geoffrey	Hinton,	2012.	
Reproduced	with	permission.	
4/3/2018

## Slide 32

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungConvolutional Neural Networks (CNN)
were not invented overnight
32 4/3/2018

## Slide 33

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
1998
2012LeCun et al.
Krizhevsky et 
al.# of transistors # of pixels used in training
# of transistors # of pixels used in training107
1014106
109
GPUs
33K
InputImage Maps
Convolutions
SubsamplingOutput
Fully Connected
Figure	copyright	Alex	 Krizhevsky ,	Ilya	
Sutskever ,	and	Geoffrey	Hinton,	2012.	
Reproduced	with	permission.	
4/3/2018

## Slide 34

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 4/3/2018 34
DataIngredients	for	 Deep	Learning
ComputationAlgorithms

## Slide 35

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 4/3/2018 35024681012141618
1/2004 10/2006 7/2009 4/2012 12/2014 9/2017 
TimeGigaFLOPs per	Dollar
CPU GPU
GeForce	
GTX	580
(AlexNet )GTX	1080	 Ti
GeForce	
8800	GTXDeep	Learning	Explosion

## Slide 36

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung0510152025303540
1/2004 10/2006 7/2009 4/2012 12/2014 9/2017 
TimeGigaFLOPs per	Dollar
CPU GPU TPU
4/3/2018 36GeForce	
GTX	580
(AlexNet )GTX	1080	 Ti
GeForce	
8800	GTXTITAN	V
(Tensor	Cores)
Deep	Learning	Explosion

## Slide 37

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungThe quest for visual intelligence 
goes far beyond object recognition…
37 4/3/2018

## Slide 38

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 38
Image is	CC0	1.0 public	domain
Laptop
Glass
DeskWall
Wire
Image is	CC	BY -SA	4.0
Image is	GFDL
Image is	CC	BY -SA	2.0
Waving
4/3/2018

## Slide 39

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 39
Johnson	 et	al. ,	“Image	Retrieval	using	Scene	Graphs”,	CVPR	2015	
Figures	copyright	IEEE,	2015.	Reproduced	for	educational	purposes
4/3/2018

## Slide 40

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungSome kind of game or fight. Two groups of two 
men? The man on the left is throwing 
something. Outdoors seemed like because i
have an impression of grass and maybe lines on 
the grass? That would be why I think perhaps a 
game, rough game though, more like rugby 
than football because they pairs weren't in 
pads and helmets, though I did get the 
impression of similar clothing. maybe some 
trees? in the background. (Subject: SM)PT = 500ms
Fei-Fei, Iyer, Koch, Perona , JoV,2007
40Image is	licensed	under	 CC	BY -SA	3.0 ;	changes	made
4/3/2018

## Slide 41

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 41This	image is	copyright -free United	States	government	work
Example	credit:	 Andrej	 Karpathy
4/3/2018

## Slide 42

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung
Computer Vision Technology
Can Better Our Lives
42Inside	four	images,	clockwise,	starting	from	top	left:
Image is	CC0	1.0 public	domain
Image by	Tucania is	licensed	under	 CC	BY -SA	3.0 ;	changes	made
Image by	Intuitive	Surgical,	Inc.	is	licensed	under	 CC	BY -SA	3.0 ;	changes	made
Image by	Oyundari Zorigtbaatar is	licensed	under	 CC	BY -SA	4.0Outside	border	images,	clockwise,	starting	from	top	left:
Image by	Pop	Culture	Geek is	licensed	under	 CC	BY	2.0 ;	changes	made
Image by	the	US	Government	is	in	the	public	domain
Image by	the	US	Government	is	in	the	public	domain
Image by	Glogger is	licensed	under	 CC	BY -SA	3.0 ;	changes	made
Image by	Sylenius is	licensed	under	 CC	BY	3.0 ;	changes	made
Image by	US	Government		is	in	the	public	domain
4/3/2018

## Slide 43

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungWho we are
43
Instructors
Teaching	Assistants
4/3/2018

## Slide 44

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungHow	to	Contact	Us
•Course	Website:	 http://cs231n.stanford.edu /
–Syllabus,	lecture	slides,	links	to	assignment	downloads,	 etc
•Piazza:	 http:// piazza.com/stanford/spring2018/cs231n
–Use	this	for	most	communication	with	course	staff
–Ask	questions	about	homework,	grading,	logistics,	 etc
–Use	private	questions	if	you	want	to	post	code
•Gradescope
–For	turning	in	homework	and	receiving	grades
•Canvas
–For	watching	lecture	videos
44 4/3/2018

## Slide 45

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungOptional	Textbook
•Deep	 Learning by	
Goodfellow ,	Bengio ,	
and	 Courville
•Free	online
45
 4/3/2018

## Slide 46

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungOur philosophy
•Thorough and Detailed.
–Understand how to write from scratch, debug 
and train convolutional neural networks.
•Practical.
–Focus on practical techniques for training these 
networks at scale, and on GPUs (e.g. will touch 
on distributed optimization, differences between 
CPU vs. GPU, etc.) Also look at state of the art 
software tools such as TensorFlow , and PyTorch
•State of the art.
–Most materials are new from research world in 
the past 1 -3 years. Very exciting stuff !
46 4/3/2018

## Slide 47

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungOur philosophy (cont’d)
•Fun. 
–Some fun topics such as Image Captioning (using RNN)
–Also DeepDream , NeuralStyle , etc.
47
 4/3/2018

## Slide 48

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungPre-requisite
•Proficiency in Python, some high -level familiarity 
with C/C++
–All class assignments will be in Python (and use 
numpy ), but some of the deep learning libraries we 
may look at later in the class are written in C++. 
–A Python tutorial available on course website
•College Calculus, Linear Algebra
•Equivalent knowledge of CS229 (Machine 
Learning)
–We will be formulating cost functions, taking 
derivatives and performing optimization with 
gradient descent.
48 4/3/2018

## Slide 49

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungGrading Policy
•3 Problem Sets: 15% x 3 = 45%
•Midterm Exam: 20%
•Course Project: 35%
–Project Proposal: 1%
–Milestone: 2% 
–Poster: 2%
–Project Report: 30%
•Late policy
–4 free late days –use up to 2 late days per assignment
–Afterwards, 25% off per day late
–No late days for project report
49 4/3/2018

## Slide 50

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungCollaboration	Policy
•We	follow	the	 Stanford	Honor	Code and	the	 CS	
Department	Honor	Code –read	them!
•Rule	1 :	Don’t	look	at	solutions	or	code	that	are	not	your	
own;	everything	you	submit	should	be	your	own	work
•Rule	2 :	Don’t	share	your	solution	code	with	others;	
however	discussing	ideas	or	general	strategies	is	fine	and	
encouraged
•Rule	3 :	Indicate	in	your	submissions	anyone	you	worked	
with
•Turning	in	something	late	/	incomplete	is	better	than	
violating	the	honor	code
4/3/2018 50

## Slide 51

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena YeungNext	Time:	Image	Classification
4/3/2018 51
K-Nearest	Neighbor
 Linear	Classifier

## Slide 52

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 52•Hubel, David H.,and Torsten N.Wiesel ."Receptive fields, binocular interaction and functional
architecture inthecat's visual cortex ."TheJournal ofphysiology 160.1(1962 ):106.[PDF]
•Roberts, Lawrence Gilman ."Machine Perception ofThree -dimensional Solids ."Diss.Massachusetts
Institute ofTechnology, 1963 .[PDF]
•Marr, David ."Vision .”TheMITPress, 1982 .[PDF]
•Brooks, Rodney A.,andCreiner, Russell andBinford, Thomas O."The ACRONYM model -based vision
system ."InProceedings ofthe6thInternational Joint Conference onArtificial Intelligence (1979 ):105-
113.[PDF]
•Fischler, Martin A.,andRobert A.Elschlager ."The representation andmatching ofpictorial structures ."
IEEE Transactions onComputers 22.1(1973 ):67-92.[PDF]
•Lowe, David G.,"Three -dimensional object recognition from single two-dimensional images," Artificial
Intelligence, 31,3(1987 ),pp.355-395.[PDF]
•Shi, Jianbo, and Jitendra Malik ."Normalized cuts and image segmentation ."Pattern Analysis and
Machine Intelligence, IEEE Transactions on22.8(2000 ):888-905.[PDF]
•Viola, Paul, andMichael Jones ."Rapid object detection using aboosted cascade ofsimple features ."
Computer Vision andPattern Recognition, 2001 .CVPR 2001 .Proceedings ofthe2001 IEEE Computer
Society Conference on.Vol.1.IEEE, 2001 .[PDF]
•Lowe, David G."Distinctive image features from scale -invariant keypoints ."International Journal of
Computer Vision 60.2(2004 ):91-110.[PDF]
•Lazebnik ,Svetlana, Cordelia Schmid ,and Jean Ponce ."Beyond bags offeatures :Spatial pyramid
matching forrecognizing natural scene categories ."Computer Vision and Pattern Recognition, 2006
IEEE Computer Society Conference on.Vol.2.IEEE, 2006 .[PDF]References
4/3/2018

## Slide 53

Lecture 1 - Fei-Fei Li & Justin Johnson & Serena Yeung 53•Dalal ,Navneet, and BillTriggs ."Histograms oforiented gradients forhuman detection ."Computer
Vision andPattern Recognition, 2005 .CVPR 2005 .IEEE Computer Society Conference on.Vol.1.IEEE,
2005 .[PDF]
•Felzenszwalb, Pedro, David McAllester, and Deva Ramanan ."Adiscriminatively trained, multiscale,
deformable part model ."Computer Vision and Pattern Recognition, 2008 .CVPR 2008 .IEEE
Conference on.IEEE, 2008 [PDF]
•Everingham, Mark, etal."The pascal visual object classes (VOC) challenge ."International Journal of
Computer Vision 88.2(2010 ):303-338.[PDF]
•Deng, Jia,etal."Imagenet :Alarge -scale hierarchical image database ."Computer Vision andPattern
Recognition, 2009 .CVPR 2009 .IEEE Conference on.IEEE, 2009 .[PDF]
•Russakovsky ,Olga, etal."Imagenet Large Scale Visual Recognition Challenge ."arXiv :1409 .0575 .[PDF]
•Lin, Yuanqing, etal."Large -scale image classification :fast feature extraction and SVM training ."
Computer Vision andPattern Recognition (CVPR), 2011 IEEE Conference on.IEEE, 2011 .[PDF]
•Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E.Hinton ."Imagenet classification with deep
convolutional neural networks ."Advances inneural information processing systems .2012 .[PDF]
•Szegedy ,Christian, etal."Going deeper with convolutions ."arXiv preprint arXiv :1409 .4842 (2014 ).
[PDF]
•Simonyan ,Karen, and Andrew Zisserman ."Very deep convolutional networks forlarge -scale image
recognition ."arXiv preprint arXiv :1409 .1556 (2014 ).[PDF]
•He,Kaiming ,etal."Spatial Pyramid Pooling inDeep Convolutional Networks forVisual Recognition ."
arXiv preprint arXiv :1406 .4729 (2014 ).[PDF]
•LeCun ,Yann ,etal."Gradient -based learning applied todocument recognition ."Proceedings ofthe
IEEE 86.11(1998 ):2278 -2324 .[PDF]
•Fei-Fei,Li,etal."What doweperceive inaglance ofareal-world scene? ."Journal ofvision 7.1(2007 ):
10.[PDF]
4/3/2018

