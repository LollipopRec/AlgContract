˪
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
?
doc_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_namedoc_embedding/embeddings
?
,doc_embedding/embeddings/Read/ReadVariableOpReadVariableOpdoc_embedding/embeddings*
_output_shapes

:@*
dtype0
?
user_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_nameuser_embedding/embeddings
?
-user_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_embedding/embeddings*
_output_shapes

:@*
dtype0
x
doc_fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedoc_fc1/kernel
q
"doc_fc1/kernel/Read/ReadVariableOpReadVariableOpdoc_fc1/kernel*
_output_shapes

:@@*
dtype0
p
doc_fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedoc_fc1/bias
i
 doc_fc1/bias/Read/ReadVariableOpReadVariableOpdoc_fc1/bias*
_output_shapes
:@*
dtype0
z
user_fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_nameuser_fc1/kernel
s
#user_fc1/kernel/Read/ReadVariableOpReadVariableOpuser_fc1/kernel*
_output_shapes

:@@*
dtype0
r
user_fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameuser_fc1/bias
k
!user_fc1/bias/Read/ReadVariableOpReadVariableOpuser_fc1/bias*
_output_shapes
:@*
dtype0
x
doc_fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedoc_fc2/kernel
q
"doc_fc2/kernel/Read/ReadVariableOpReadVariableOpdoc_fc2/kernel*
_output_shapes

:@@*
dtype0
p
doc_fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedoc_fc2/bias
i
 doc_fc2/bias/Read/ReadVariableOpReadVariableOpdoc_fc2/bias*
_output_shapes
:@*
dtype0
z
user_fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_nameuser_fc2/kernel
s
#user_fc2/kernel/Read/ReadVariableOpReadVariableOpuser_fc2/kernel*
_output_shapes

:@@*
dtype0
r
user_fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameuser_fc2/bias
k
!user_fc2/bias/Read/ReadVariableOpReadVariableOpuser_fc2/bias*
_output_shapes
:@*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/doc_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/doc_embedding/embeddings/m
?
3Adam/doc_embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/doc_embedding/embeddings/m*
_output_shapes

:@*
dtype0
?
 Adam/user_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" Adam/user_embedding/embeddings/m
?
4Adam/user_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/m*
_output_shapes

:@*
dtype0
?
Adam/doc_fc1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/doc_fc1/kernel/m

)Adam/doc_fc1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/doc_fc1/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/doc_fc1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/doc_fc1/bias/m
w
'Adam/doc_fc1/bias/m/Read/ReadVariableOpReadVariableOpAdam/doc_fc1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/user_fc1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/user_fc1/kernel/m
?
*Adam/user_fc1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/user_fc1/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/user_fc1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/user_fc1/bias/m
y
(Adam/user_fc1/bias/m/Read/ReadVariableOpReadVariableOpAdam/user_fc1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/doc_fc2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/doc_fc2/kernel/m

)Adam/doc_fc2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/doc_fc2/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/doc_fc2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/doc_fc2/bias/m
w
'Adam/doc_fc2/bias/m/Read/ReadVariableOpReadVariableOpAdam/doc_fc2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/user_fc2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/user_fc2/kernel/m
?
*Adam/user_fc2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/user_fc2/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/user_fc2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/user_fc2/bias/m
y
(Adam/user_fc2/bias/m/Read/ReadVariableOpReadVariableOpAdam/user_fc2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/doc_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/doc_embedding/embeddings/v
?
3Adam/doc_embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/doc_embedding/embeddings/v*
_output_shapes

:@*
dtype0
?
 Adam/user_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*1
shared_name" Adam/user_embedding/embeddings/v
?
4Adam/user_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/v*
_output_shapes

:@*
dtype0
?
Adam/doc_fc1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/doc_fc1/kernel/v

)Adam/doc_fc1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/doc_fc1/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/doc_fc1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/doc_fc1/bias/v
w
'Adam/doc_fc1/bias/v/Read/ReadVariableOpReadVariableOpAdam/doc_fc1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/user_fc1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/user_fc1/kernel/v
?
*Adam/user_fc1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/user_fc1/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/user_fc1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/user_fc1/bias/v
y
(Adam/user_fc1/bias/v/Read/ReadVariableOpReadVariableOpAdam/user_fc1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/doc_fc2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/doc_fc2/kernel/v

)Adam/doc_fc2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/doc_fc2/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/doc_fc2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/doc_fc2/bias/v
w
'Adam/doc_fc2/bias/v/Read/ReadVariableOpReadVariableOpAdam/doc_fc2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/user_fc2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/user_fc2/kernel/v
?
*Adam/user_fc2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/user_fc2/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/user_fc2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/user_fc2/bias/v
y
(Adam/user_fc2/bias/v/Read/ReadVariableOpReadVariableOpAdam/user_fc2/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
	optimizer

signatures
trainable_variables
regularization_losses
	variables
	keras_api
 
 
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
R
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
h

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
R
[trainable_variables
\regularization_losses
]	variables
^	keras_api
R
_trainable_variables
`regularization_losses
a	variables
b	keras_api
?
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem?m?#m?$m?)m?*m?Om?Pm?Um?Vm?v?v?#v?$v?)v?*v?Ov?Pv?Uv?Vv?
 
F
0
1
#2
$3
)4
*5
O6
P7
U8
V9
 
F
0
1
#2
$3
)4
*5
O6
P7
U8
V9
?
hmetrics

ilayers
jlayer_regularization_losses
trainable_variables
regularization_losses
klayer_metrics
	variables
lnon_trainable_variables
hf
VARIABLE_VALUEdoc_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
mmetrics

nlayers
olayer_regularization_losses
trainable_variables
regularization_losses
player_metrics
	variables
qnon_trainable_variables
ig
VARIABLE_VALUEuser_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
rmetrics

slayers
tlayer_regularization_losses
trainable_variables
 regularization_losses
ulayer_metrics
!	variables
vnon_trainable_variables
ZX
VARIABLE_VALUEdoc_fc1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdoc_fc1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
?
wmetrics

xlayers
ylayer_regularization_losses
%trainable_variables
&regularization_losses
zlayer_metrics
'	variables
{non_trainable_variables
[Y
VARIABLE_VALUEuser_fc1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEuser_fc1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?
|metrics

}layers
~layer_regularization_losses
+trainable_variables
,regularization_losses
layer_metrics
-	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
/trainable_variables
0regularization_losses
?layer_metrics
1	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?layer_metrics
5	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
7trainable_variables
8regularization_losses
?layer_metrics
9	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
;trainable_variables
<regularization_losses
?layer_metrics
=	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
?trainable_variables
@regularization_losses
?layer_metrics
A	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
Ctrainable_variables
Dregularization_losses
?layer_metrics
E	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
Gtrainable_variables
Hregularization_losses
?layer_metrics
I	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
Ktrainable_variables
Lregularization_losses
?layer_metrics
M	variables
?non_trainable_variables
ZX
VARIABLE_VALUEdoc_fc2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdoc_fc2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
?
?metrics
?layers
 ?layer_regularization_losses
Qtrainable_variables
Rregularization_losses
?layer_metrics
S	variables
?non_trainable_variables
[Y
VARIABLE_VALUEuser_fc2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEuser_fc2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
?
?metrics
?layers
 ?layer_regularization_losses
Wtrainable_variables
Xregularization_losses
?layer_metrics
Y	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
[trainable_variables
\regularization_losses
?layer_metrics
]	variables
?non_trainable_variables
 
 
 
?
?metrics
?layers
 ?layer_regularization_losses
_trainable_variables
`regularization_losses
?layer_metrics
a	variables
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUEAdam/doc_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/user_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/doc_fc1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/doc_fc1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/user_fc1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/user_fc1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/doc_fc2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/doc_fc2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/user_fc2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/user_fc2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/doc_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/user_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/doc_fc1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/doc_fc1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/user_fc1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/user_fc1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/doc_fc2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/doc_fc2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/user_fc2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/user_fc2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_doc_idsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

{
serving_default_user_idsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_doc_idsserving_default_user_idsuser_embedding/embeddingsdoc_embedding/embeddingsuser_fc1/kerneluser_fc1/biasdoc_fc1/kerneldoc_fc1/biasdoc_fc2/kerneldoc_fc2/biasuser_fc2/kerneluser_fc2/bias*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_3788
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,doc_embedding/embeddings/Read/ReadVariableOp-user_embedding/embeddings/Read/ReadVariableOp"doc_fc1/kernel/Read/ReadVariableOp doc_fc1/bias/Read/ReadVariableOp#user_fc1/kernel/Read/ReadVariableOp!user_fc1/bias/Read/ReadVariableOp"doc_fc2/kernel/Read/ReadVariableOp doc_fc2/bias/Read/ReadVariableOp#user_fc2/kernel/Read/ReadVariableOp!user_fc2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/doc_embedding/embeddings/m/Read/ReadVariableOp4Adam/user_embedding/embeddings/m/Read/ReadVariableOp)Adam/doc_fc1/kernel/m/Read/ReadVariableOp'Adam/doc_fc1/bias/m/Read/ReadVariableOp*Adam/user_fc1/kernel/m/Read/ReadVariableOp(Adam/user_fc1/bias/m/Read/ReadVariableOp)Adam/doc_fc2/kernel/m/Read/ReadVariableOp'Adam/doc_fc2/bias/m/Read/ReadVariableOp*Adam/user_fc2/kernel/m/Read/ReadVariableOp(Adam/user_fc2/bias/m/Read/ReadVariableOp3Adam/doc_embedding/embeddings/v/Read/ReadVariableOp4Adam/user_embedding/embeddings/v/Read/ReadVariableOp)Adam/doc_fc1/kernel/v/Read/ReadVariableOp'Adam/doc_fc1/bias/v/Read/ReadVariableOp*Adam/user_fc1/kernel/v/Read/ReadVariableOp(Adam/user_fc1/bias/v/Read/ReadVariableOp)Adam/doc_fc2/kernel/v/Read/ReadVariableOp'Adam/doc_fc2/bias/v/Read/ReadVariableOp*Adam/user_fc2/kernel/v/Read/ReadVariableOp(Adam/user_fc2/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_4434
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedoc_embedding/embeddingsuser_embedding/embeddingsdoc_fc1/kerneldoc_fc1/biasuser_fc1/kerneluser_fc1/biasdoc_fc2/kerneldoc_fc2/biasuser_fc2/kerneluser_fc2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/doc_embedding/embeddings/m Adam/user_embedding/embeddings/mAdam/doc_fc1/kernel/mAdam/doc_fc1/bias/mAdam/user_fc1/kernel/mAdam/user_fc1/bias/mAdam/doc_fc2/kernel/mAdam/doc_fc2/bias/mAdam/user_fc2/kernel/mAdam/user_fc2/bias/mAdam/doc_embedding/embeddings/v Adam/user_embedding/embeddings/vAdam/doc_fc1/kernel/vAdam/doc_fc1/bias/vAdam/user_fc1/kernel/vAdam/user_fc1/bias/vAdam/doc_fc2/kernel/vAdam/doc_fc2/bias/vAdam/user_fc2/kernel/vAdam/user_fc2/bias/v*1
Tin*
(2&*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_4557??

?
?
$__inference_model_layer_call_fn_3752
doc_ids
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsuser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_37292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
N
2__inference_tf_op_layer_MaxPool_layer_call_fn_4182

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_34322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
?
H__inference_user_embedding_layer_call_and_return_conditional_losses_4063

inputs
embedding_lookup_4057
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_4057Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/4057*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/4057*+
_output_shapes
:?????????
@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
S
7__inference_tf_op_layer_ExpandDims_1_layer_call_fn_4172

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_33922
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3617
doc_ids
user_ids
user_embedding_3580
doc_embedding_3583
user_fc1_3586
user_fc1_3588
doc_fc1_3591
doc_fc1_3593
doc_fc2_3604
doc_fc2_3606
user_fc2_3609
user_fc2_3611
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_idsuser_embedding_3580*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_user_embedding_layer_call_and_return_conditional_losses_32582(
&user_embedding/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCalldoc_idsdoc_embedding_3583*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_doc_embedding_layer_call_and_return_conditional_losses_32802'
%doc_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_3586user_fc1_3588*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc1_layer_call_and_return_conditional_losses_33232"
 user_fc1/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_3591doc_fc1_3593*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc1_layer_call_and_return_conditional_losses_33702!
doc_fc1/StatefulPartitionedCall?
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_33922*
(tf_op_layer_ExpandDims_1/PartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall(doc_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_34062(
&tf_op_layer_ExpandDims/PartitionedCall?
%tf_op_layer_MaxPool_1/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_34192'
%tf_op_layer_MaxPool_1/PartitionedCall?
#tf_op_layer_MaxPool/PartitionedCallPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_34322%
#tf_op_layer_MaxPool/PartitionedCall?
%tf_op_layer_Squeeze_2/PartitionedCallPartitionedCall.tf_op_layer_MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_34452'
%tf_op_layer_Squeeze_2/PartitionedCall?
#tf_op_layer_Squeeze/PartitionedCallPartitionedCall,tf_op_layer_MaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_34582%
#tf_op_layer_Squeeze/PartitionedCall?
%tf_op_layer_Squeeze_3/PartitionedCallPartitionedCall.tf_op_layer_Squeeze_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_34712'
%tf_op_layer_Squeeze_3/PartitionedCall?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall,tf_op_layer_Squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_34842'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_3604doc_fc2_3606*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc2_layer_call_and_return_conditional_losses_35032!
doc_fc2/StatefulPartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_3/PartitionedCall:output:0user_fc2_3609user_fc2_3611*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc2_layer_call_and_return_conditional_losses_35302"
 user_fc2/StatefulPartitionedCall?
tf_op_layer_Mul/PartitionedCallPartitionedCall(doc_fc2/StatefulPartitionedCall:output:0)user_fc2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_35522!
tf_op_layer_Mul/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_35672!
tf_op_layer_Sum/PartitionedCall?
IdentityIdentity(tf_op_layer_Sum/PartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3406

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_4101

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@:::S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ƣ
?
 __inference__traced_restore_4557
file_prefix-
)assignvariableop_doc_embedding_embeddings0
,assignvariableop_1_user_embedding_embeddings%
!assignvariableop_2_doc_fc1_kernel#
assignvariableop_3_doc_fc1_bias&
"assignvariableop_4_user_fc1_kernel$
 assignvariableop_5_user_fc1_bias%
!assignvariableop_6_doc_fc2_kernel#
assignvariableop_7_doc_fc2_bias&
"assignvariableop_8_user_fc2_kernel$
 assignvariableop_9_user_fc2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count7
3assignvariableop_17_adam_doc_embedding_embeddings_m8
4assignvariableop_18_adam_user_embedding_embeddings_m-
)assignvariableop_19_adam_doc_fc1_kernel_m+
'assignvariableop_20_adam_doc_fc1_bias_m.
*assignvariableop_21_adam_user_fc1_kernel_m,
(assignvariableop_22_adam_user_fc1_bias_m-
)assignvariableop_23_adam_doc_fc2_kernel_m+
'assignvariableop_24_adam_doc_fc2_bias_m.
*assignvariableop_25_adam_user_fc2_kernel_m,
(assignvariableop_26_adam_user_fc2_bias_m7
3assignvariableop_27_adam_doc_embedding_embeddings_v8
4assignvariableop_28_adam_user_embedding_embeddings_v-
)assignvariableop_29_adam_doc_fc1_kernel_v+
'assignvariableop_30_adam_doc_fc1_bias_v.
*assignvariableop_31_adam_user_fc1_kernel_v,
(assignvariableop_32_adam_user_fc1_bias_v-
)assignvariableop_33_adam_doc_fc2_kernel_v+
'assignvariableop_34_adam_doc_fc2_bias_v.
*assignvariableop_35_adam_user_fc2_kernel_v,
(assignvariableop_36_adam_user_fc2_bias_v
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_doc_embedding_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_user_embedding_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_doc_fc1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_doc_fc1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_user_fc1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_user_fc1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_doc_fc2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_doc_fc2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_user_fc2_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_user_fc2_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_doc_embedding_embeddings_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_user_embedding_embeddings_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_doc_fc1_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_doc_fc1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_user_fc1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_user_fc1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_doc_fc2_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_doc_fc2_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_user_fc2_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_user_fc2_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_doc_embedding_embeddings_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_user_embedding_embeddings_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_doc_fc1_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_doc_fc1_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_user_fc1_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_user_fc1_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_doc_fc2_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_doc_fc2_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_user_fc2_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_user_fc2_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
?
N
2__inference_tf_op_layer_Squeeze_layer_call_fn_4202

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_34582
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
i
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_3458

inputs
identity?
SqueezeSqueezeinputs*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3576
doc_ids
user_ids
user_embedding_3267
doc_embedding_3289
user_fc1_3334
user_fc1_3336
doc_fc1_3381
doc_fc1_3383
doc_fc2_3514
doc_fc2_3516
user_fc2_3541
user_fc2_3543
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_idsuser_embedding_3267*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_user_embedding_layer_call_and_return_conditional_losses_32582(
&user_embedding/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCalldoc_idsdoc_embedding_3289*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_doc_embedding_layer_call_and_return_conditional_losses_32802'
%doc_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_3334user_fc1_3336*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc1_layer_call_and_return_conditional_losses_33232"
 user_fc1/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_3381doc_fc1_3383*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc1_layer_call_and_return_conditional_losses_33702!
doc_fc1/StatefulPartitionedCall?
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_33922*
(tf_op_layer_ExpandDims_1/PartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall(doc_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_34062(
&tf_op_layer_ExpandDims/PartitionedCall?
%tf_op_layer_MaxPool_1/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_34192'
%tf_op_layer_MaxPool_1/PartitionedCall?
#tf_op_layer_MaxPool/PartitionedCallPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_34322%
#tf_op_layer_MaxPool/PartitionedCall?
%tf_op_layer_Squeeze_2/PartitionedCallPartitionedCall.tf_op_layer_MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_34452'
%tf_op_layer_Squeeze_2/PartitionedCall?
#tf_op_layer_Squeeze/PartitionedCallPartitionedCall,tf_op_layer_MaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_34582%
#tf_op_layer_Squeeze/PartitionedCall?
%tf_op_layer_Squeeze_3/PartitionedCallPartitionedCall.tf_op_layer_Squeeze_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_34712'
%tf_op_layer_Squeeze_3/PartitionedCall?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall,tf_op_layer_Squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_34842'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_3514doc_fc2_3516*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc2_layer_call_and_return_conditional_losses_35032!
doc_fc2/StatefulPartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_3/PartitionedCall:output:0user_fc2_3541user_fc2_3543*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc2_layer_call_and_return_conditional_losses_35302"
 user_fc2/StatefulPartitionedCall?
tf_op_layer_Mul/PartitionedCallPartitionedCall(doc_fc2/StatefulPartitionedCall:output:0)user_fc2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_35522!
tf_op_layer_Mul/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_35672!
tf_op_layer_Sum/PartitionedCall?
IdentityIdentity(tf_op_layer_Sum/PartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_3419

inputs
identity?
	MaxPool_1MaxPoolinputs*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2
	MaxPool_1n
IdentityIdentityMaxPool_1:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
?
G__inference_doc_embedding_layer_call_and_return_conditional_losses_3280

inputs
embedding_lookup_3274
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_3274Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/3274*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/3274*+
_output_shapes
:?????????
@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
?
B__inference_user_fc1_layer_call_and_return_conditional_losses_3323

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@:::S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_user_fc2_layer_call_fn_4272

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc2_layer_call_and_return_conditional_losses_35302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_4207

inputs
identity?
	Squeeze_2Squeezeinputs*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_2j
IdentityIdentitySqueeze_2:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_3484

inputs
identity?
	Squeeze_1Squeezeinputs*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_1f
IdentityIdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_tf_op_layer_Sum_layer_call_fn_4295

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_35672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_user_fc2_layer_call_and_return_conditional_losses_3530

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_4217

inputs
identity?
	Squeeze_1Squeezeinputs*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_1f
IdentityIdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_4243

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
i
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_4197

inputs
identity?
SqueezeSqueezeinputs*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
|
'__inference_user_fc1_layer_call_fn_4150

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc1_layer_call_and_return_conditional_losses_33232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4290

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
r
,__inference_doc_embedding_layer_call_fn_4053

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_doc_embedding_layer_call_and_return_conditional_losses_32802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_3503

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_3471

inputs
identity?
	Squeeze_3Squeezeinputs*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_3f
IdentityIdentitySqueeze_3:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_3370

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@:::S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?W
?
__inference__traced_save_4434
file_prefix7
3savev2_doc_embedding_embeddings_read_readvariableop8
4savev2_user_embedding_embeddings_read_readvariableop-
)savev2_doc_fc1_kernel_read_readvariableop+
'savev2_doc_fc1_bias_read_readvariableop.
*savev2_user_fc1_kernel_read_readvariableop,
(savev2_user_fc1_bias_read_readvariableop-
)savev2_doc_fc2_kernel_read_readvariableop+
'savev2_doc_fc2_bias_read_readvariableop.
*savev2_user_fc2_kernel_read_readvariableop,
(savev2_user_fc2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_doc_embedding_embeddings_m_read_readvariableop?
;savev2_adam_user_embedding_embeddings_m_read_readvariableop4
0savev2_adam_doc_fc1_kernel_m_read_readvariableop2
.savev2_adam_doc_fc1_bias_m_read_readvariableop5
1savev2_adam_user_fc1_kernel_m_read_readvariableop3
/savev2_adam_user_fc1_bias_m_read_readvariableop4
0savev2_adam_doc_fc2_kernel_m_read_readvariableop2
.savev2_adam_doc_fc2_bias_m_read_readvariableop5
1savev2_adam_user_fc2_kernel_m_read_readvariableop3
/savev2_adam_user_fc2_bias_m_read_readvariableop>
:savev2_adam_doc_embedding_embeddings_v_read_readvariableop?
;savev2_adam_user_embedding_embeddings_v_read_readvariableop4
0savev2_adam_doc_fc1_kernel_v_read_readvariableop2
.savev2_adam_doc_fc1_bias_v_read_readvariableop5
1savev2_adam_user_fc1_kernel_v_read_readvariableop3
/savev2_adam_user_fc1_bias_v_read_readvariableop4
0savev2_adam_doc_fc2_kernel_v_read_readvariableop2
.savev2_adam_doc_fc2_bias_v_read_readvariableop5
1savev2_adam_user_fc2_kernel_v_read_readvariableop3
/savev2_adam_user_fc2_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bfa6ff65cb304594b49c2ef3594c56e6/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*?
value?B?%B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_doc_embedding_embeddings_read_readvariableop4savev2_user_embedding_embeddings_read_readvariableop)savev2_doc_fc1_kernel_read_readvariableop'savev2_doc_fc1_bias_read_readvariableop*savev2_user_fc1_kernel_read_readvariableop(savev2_user_fc1_bias_read_readvariableop)savev2_doc_fc2_kernel_read_readvariableop'savev2_doc_fc2_bias_read_readvariableop*savev2_user_fc2_kernel_read_readvariableop(savev2_user_fc2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_doc_embedding_embeddings_m_read_readvariableop;savev2_adam_user_embedding_embeddings_m_read_readvariableop0savev2_adam_doc_fc1_kernel_m_read_readvariableop.savev2_adam_doc_fc1_bias_m_read_readvariableop1savev2_adam_user_fc1_kernel_m_read_readvariableop/savev2_adam_user_fc1_bias_m_read_readvariableop0savev2_adam_doc_fc2_kernel_m_read_readvariableop.savev2_adam_doc_fc2_bias_m_read_readvariableop1savev2_adam_user_fc2_kernel_m_read_readvariableop/savev2_adam_user_fc2_bias_m_read_readvariableop:savev2_adam_doc_embedding_embeddings_v_read_readvariableop;savev2_adam_user_embedding_embeddings_v_read_readvariableop0savev2_adam_doc_fc1_kernel_v_read_readvariableop.savev2_adam_doc_fc1_bias_v_read_readvariableop1savev2_adam_user_fc1_kernel_v_read_readvariableop/savev2_adam_user_fc1_bias_v_read_readvariableop0savev2_adam_doc_fc2_kernel_v_read_readvariableop.savev2_adam_doc_fc2_bias_v_read_readvariableop1savev2_adam_user_fc2_kernel_v_read_readvariableop/savev2_adam_user_fc2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@@:@:@@:@:@@:@: : : : : : : :@:@:@@:@:@@:@:@@:@:@@:@:@:@:@@:@:@@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$  

_output_shapes

:@@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@:&

_output_shapes
: 
?
n
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_4167

inputs
identityf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputsExpandDims_1/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2
ExpandDims_1q
IdentityIdentityExpandDims_1:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_3243
doc_ids
user_ids.
*model_user_embedding_embedding_lookup_3150-
)model_doc_embedding_embedding_lookup_31564
0model_user_fc1_tensordot_readvariableop_resource2
.model_user_fc1_biasadd_readvariableop_resource3
/model_doc_fc1_tensordot_readvariableop_resource1
-model_doc_fc1_biasadd_readvariableop_resource0
,model_doc_fc2_matmul_readvariableop_resource1
-model_doc_fc2_biasadd_readvariableop_resource1
-model_user_fc2_matmul_readvariableop_resource2
.model_user_fc2_biasadd_readvariableop_resource
identity??
model/user_embedding/CastCastuser_ids*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
model/user_embedding/Cast?
%model/user_embedding/embedding_lookupResourceGather*model_user_embedding_embedding_lookup_3150model/user_embedding/Cast:y:0*
Tindices0*=
_class3
1/loc:@model/user_embedding/embedding_lookup/3150*+
_output_shapes
:?????????
@*
dtype02'
%model/user_embedding/embedding_lookup?
.model/user_embedding/embedding_lookup/IdentityIdentity.model/user_embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model/user_embedding/embedding_lookup/3150*+
_output_shapes
:?????????
@20
.model/user_embedding/embedding_lookup/Identity?
0model/user_embedding/embedding_lookup/Identity_1Identity7model/user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@22
0model/user_embedding/embedding_lookup/Identity_1?
model/doc_embedding/CastCastdoc_ids*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
model/doc_embedding/Cast?
$model/doc_embedding/embedding_lookupResourceGather)model_doc_embedding_embedding_lookup_3156model/doc_embedding/Cast:y:0*
Tindices0*<
_class2
0.loc:@model/doc_embedding/embedding_lookup/3156*+
_output_shapes
:?????????
@*
dtype02&
$model/doc_embedding/embedding_lookup?
-model/doc_embedding/embedding_lookup/IdentityIdentity-model/doc_embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@model/doc_embedding/embedding_lookup/3156*+
_output_shapes
:?????????
@2/
-model/doc_embedding/embedding_lookup/Identity?
/model/doc_embedding/embedding_lookup/Identity_1Identity6model/doc_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@21
/model/doc_embedding/embedding_lookup/Identity_1?
'model/user_fc1/Tensordot/ReadVariableOpReadVariableOp0model_user_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'model/user_fc1/Tensordot/ReadVariableOp?
model/user_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model/user_fc1/Tensordot/axes?
model/user_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model/user_fc1/Tensordot/free?
model/user_fc1/Tensordot/ShapeShape9model/user_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2 
model/user_fc1/Tensordot/Shape?
&model/user_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/user_fc1/Tensordot/GatherV2/axis?
!model/user_fc1/Tensordot/GatherV2GatherV2'model/user_fc1/Tensordot/Shape:output:0&model/user_fc1/Tensordot/free:output:0/model/user_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!model/user_fc1/Tensordot/GatherV2?
(model/user_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model/user_fc1/Tensordot/GatherV2_1/axis?
#model/user_fc1/Tensordot/GatherV2_1GatherV2'model/user_fc1/Tensordot/Shape:output:0&model/user_fc1/Tensordot/axes:output:01model/user_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#model/user_fc1/Tensordot/GatherV2_1?
model/user_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
model/user_fc1/Tensordot/Const?
model/user_fc1/Tensordot/ProdProd*model/user_fc1/Tensordot/GatherV2:output:0'model/user_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model/user_fc1/Tensordot/Prod?
 model/user_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 model/user_fc1/Tensordot/Const_1?
model/user_fc1/Tensordot/Prod_1Prod,model/user_fc1/Tensordot/GatherV2_1:output:0)model/user_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2!
model/user_fc1/Tensordot/Prod_1?
$model/user_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/user_fc1/Tensordot/concat/axis?
model/user_fc1/Tensordot/concatConcatV2&model/user_fc1/Tensordot/free:output:0&model/user_fc1/Tensordot/axes:output:0-model/user_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
model/user_fc1/Tensordot/concat?
model/user_fc1/Tensordot/stackPack&model/user_fc1/Tensordot/Prod:output:0(model/user_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2 
model/user_fc1/Tensordot/stack?
"model/user_fc1/Tensordot/transpose	Transpose9model/user_embedding/embedding_lookup/Identity_1:output:0(model/user_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2$
"model/user_fc1/Tensordot/transpose?
 model/user_fc1/Tensordot/ReshapeReshape&model/user_fc1/Tensordot/transpose:y:0'model/user_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2"
 model/user_fc1/Tensordot/Reshape?
model/user_fc1/Tensordot/MatMulMatMul)model/user_fc1/Tensordot/Reshape:output:0/model/user_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
model/user_fc1/Tensordot/MatMul?
 model/user_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2"
 model/user_fc1/Tensordot/Const_2?
&model/user_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/user_fc1/Tensordot/concat_1/axis?
!model/user_fc1/Tensordot/concat_1ConcatV2*model/user_fc1/Tensordot/GatherV2:output:0)model/user_fc1/Tensordot/Const_2:output:0/model/user_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2#
!model/user_fc1/Tensordot/concat_1?
model/user_fc1/TensordotReshape)model/user_fc1/Tensordot/MatMul:product:0*model/user_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
model/user_fc1/Tensordot?
%model/user_fc1/BiasAdd/ReadVariableOpReadVariableOp.model_user_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/user_fc1/BiasAdd/ReadVariableOp?
model/user_fc1/BiasAddBiasAdd!model/user_fc1/Tensordot:output:0-model/user_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
model/user_fc1/BiasAdd?
model/user_fc1/ReluRelumodel/user_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
model/user_fc1/Relu?
&model/doc_fc1/Tensordot/ReadVariableOpReadVariableOp/model_doc_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model/doc_fc1/Tensordot/ReadVariableOp?
model/doc_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model/doc_fc1/Tensordot/axes?
model/doc_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model/doc_fc1/Tensordot/free?
model/doc_fc1/Tensordot/ShapeShape8model/doc_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
model/doc_fc1/Tensordot/Shape?
%model/doc_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/doc_fc1/Tensordot/GatherV2/axis?
 model/doc_fc1/Tensordot/GatherV2GatherV2&model/doc_fc1/Tensordot/Shape:output:0%model/doc_fc1/Tensordot/free:output:0.model/doc_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 model/doc_fc1/Tensordot/GatherV2?
'model/doc_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model/doc_fc1/Tensordot/GatherV2_1/axis?
"model/doc_fc1/Tensordot/GatherV2_1GatherV2&model/doc_fc1/Tensordot/Shape:output:0%model/doc_fc1/Tensordot/axes:output:00model/doc_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model/doc_fc1/Tensordot/GatherV2_1?
model/doc_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
model/doc_fc1/Tensordot/Const?
model/doc_fc1/Tensordot/ProdProd)model/doc_fc1/Tensordot/GatherV2:output:0&model/doc_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model/doc_fc1/Tensordot/Prod?
model/doc_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
model/doc_fc1/Tensordot/Const_1?
model/doc_fc1/Tensordot/Prod_1Prod+model/doc_fc1/Tensordot/GatherV2_1:output:0(model/doc_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
model/doc_fc1/Tensordot/Prod_1?
#model/doc_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/doc_fc1/Tensordot/concat/axis?
model/doc_fc1/Tensordot/concatConcatV2%model/doc_fc1/Tensordot/free:output:0%model/doc_fc1/Tensordot/axes:output:0,model/doc_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
model/doc_fc1/Tensordot/concat?
model/doc_fc1/Tensordot/stackPack%model/doc_fc1/Tensordot/Prod:output:0'model/doc_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
model/doc_fc1/Tensordot/stack?
!model/doc_fc1/Tensordot/transpose	Transpose8model/doc_embedding/embedding_lookup/Identity_1:output:0'model/doc_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2#
!model/doc_fc1/Tensordot/transpose?
model/doc_fc1/Tensordot/ReshapeReshape%model/doc_fc1/Tensordot/transpose:y:0&model/doc_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
model/doc_fc1/Tensordot/Reshape?
model/doc_fc1/Tensordot/MatMulMatMul(model/doc_fc1/Tensordot/Reshape:output:0.model/doc_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
model/doc_fc1/Tensordot/MatMul?
model/doc_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2!
model/doc_fc1/Tensordot/Const_2?
%model/doc_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/doc_fc1/Tensordot/concat_1/axis?
 model/doc_fc1/Tensordot/concat_1ConcatV2)model/doc_fc1/Tensordot/GatherV2:output:0(model/doc_fc1/Tensordot/Const_2:output:0.model/doc_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 model/doc_fc1/Tensordot/concat_1?
model/doc_fc1/TensordotReshape(model/doc_fc1/Tensordot/MatMul:product:0)model/doc_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
model/doc_fc1/Tensordot?
$model/doc_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_doc_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/doc_fc1/BiasAdd/ReadVariableOp?
model/doc_fc1/BiasAddBiasAdd model/doc_fc1/Tensordot:output:0,model/doc_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
model/doc_fc1/BiasAdd?
model/doc_fc1/ReluRelumodel/doc_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
model/doc_fc1/Relu?
/model/tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :21
/model/tf_op_layer_ExpandDims_1/ExpandDims_1/dim?
+model/tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDims!model/user_fc1/Relu:activations:08model/tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2-
+model/tf_op_layer_ExpandDims_1/ExpandDims_1?
+model/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/tf_op_layer_ExpandDims/ExpandDims/dim?
'model/tf_op_layer_ExpandDims/ExpandDims
ExpandDims model/doc_fc1/Relu:activations:04model/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2)
'model/tf_op_layer_ExpandDims/ExpandDims?
%model/tf_op_layer_MaxPool_1/MaxPool_1MaxPool4model/tf_op_layer_ExpandDims_1/ExpandDims_1:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2'
%model/tf_op_layer_MaxPool_1/MaxPool_1?
!model/tf_op_layer_MaxPool/MaxPoolMaxPool0model/tf_op_layer_ExpandDims/ExpandDims:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2#
!model/tf_op_layer_MaxPool/MaxPool?
%model/tf_op_layer_Squeeze_2/Squeeze_2Squeeze.model/tf_op_layer_MaxPool_1/MaxPool_1:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2'
%model/tf_op_layer_Squeeze_2/Squeeze_2?
!model/tf_op_layer_Squeeze/SqueezeSqueeze*model/tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2#
!model/tf_op_layer_Squeeze/Squeeze?
%model/tf_op_layer_Squeeze_3/Squeeze_3Squeeze.model/tf_op_layer_Squeeze_2/Squeeze_2:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2'
%model/tf_op_layer_Squeeze_3/Squeeze_3?
%model/tf_op_layer_Squeeze_1/Squeeze_1Squeeze*model/tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2'
%model/tf_op_layer_Squeeze_1/Squeeze_1?
#model/doc_fc2/MatMul/ReadVariableOpReadVariableOp,model_doc_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model/doc_fc2/MatMul/ReadVariableOp?
model/doc_fc2/MatMulMatMul.model/tf_op_layer_Squeeze_1/Squeeze_1:output:0+model/doc_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/doc_fc2/MatMul?
$model/doc_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_doc_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/doc_fc2/BiasAdd/ReadVariableOp?
model/doc_fc2/BiasAddBiasAddmodel/doc_fc2/MatMul:product:0,model/doc_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/doc_fc2/BiasAdd?
model/doc_fc2/ReluRelumodel/doc_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/doc_fc2/Relu?
$model/user_fc2/MatMul/ReadVariableOpReadVariableOp-model_user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$model/user_fc2/MatMul/ReadVariableOp?
model/user_fc2/MatMulMatMul.model/tf_op_layer_Squeeze_3/Squeeze_3:output:0,model/user_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/user_fc2/MatMul?
%model/user_fc2/BiasAdd/ReadVariableOpReadVariableOp.model_user_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/user_fc2/BiasAdd/ReadVariableOp?
model/user_fc2/BiasAddBiasAddmodel/user_fc2/MatMul:product:0-model/user_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/user_fc2/BiasAdd?
model/user_fc2/ReluRelumodel/user_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/user_fc2/Relu?
model/tf_op_layer_Mul/MulMul model/doc_fc2/Relu:activations:0!model/user_fc2/Relu:activations:0*
T0*
_cloned(*'
_output_shapes
:?????????@2
model/tf_op_layer_Mul/Mul?
+model/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/tf_op_layer_Sum/Sum/reduction_indices?
model/tf_op_layer_Sum/SumSummodel/tf_op_layer_Mul/Mul:z:04model/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
model/tf_op_layer_Sum/Sumv
IdentityIdentity"model/tf_op_layer_Sum/Sum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
:::::::::::P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
$__inference_model_layer_call_fn_4010
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_36622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
k
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_3445

inputs
identity?
	Squeeze_2Squeezeinputs*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_2j
IdentityIdentitySqueeze_2:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
P
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_4222

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_34842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_3662

inputs
inputs_1
user_embedding_3625
doc_embedding_3628
user_fc1_3631
user_fc1_3633
doc_fc1_3636
doc_fc1_3638
doc_fc2_3649
doc_fc2_3651
user_fc2_3654
user_fc2_3656
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_embedding_3625*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_user_embedding_layer_call_and_return_conditional_losses_32582(
&user_embedding/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsdoc_embedding_3628*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_doc_embedding_layer_call_and_return_conditional_losses_32802'
%doc_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_3631user_fc1_3633*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc1_layer_call_and_return_conditional_losses_33232"
 user_fc1/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_3636doc_fc1_3638*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc1_layer_call_and_return_conditional_losses_33702!
doc_fc1/StatefulPartitionedCall?
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_33922*
(tf_op_layer_ExpandDims_1/PartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall(doc_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_34062(
&tf_op_layer_ExpandDims/PartitionedCall?
%tf_op_layer_MaxPool_1/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_34192'
%tf_op_layer_MaxPool_1/PartitionedCall?
#tf_op_layer_MaxPool/PartitionedCallPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_34322%
#tf_op_layer_MaxPool/PartitionedCall?
%tf_op_layer_Squeeze_2/PartitionedCallPartitionedCall.tf_op_layer_MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_34452'
%tf_op_layer_Squeeze_2/PartitionedCall?
#tf_op_layer_Squeeze/PartitionedCallPartitionedCall,tf_op_layer_MaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_34582%
#tf_op_layer_Squeeze/PartitionedCall?
%tf_op_layer_Squeeze_3/PartitionedCallPartitionedCall.tf_op_layer_Squeeze_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_34712'
%tf_op_layer_Squeeze_3/PartitionedCall?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall,tf_op_layer_Squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_34842'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_3649doc_fc2_3651*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc2_layer_call_and_return_conditional_losses_35032!
doc_fc2/StatefulPartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_3/PartitionedCall:output:0user_fc2_3654user_fc2_3656*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc2_layer_call_and_return_conditional_losses_35302"
 user_fc2/StatefulPartitionedCall?
tf_op_layer_Mul/PartitionedCallPartitionedCall(doc_fc2/StatefulPartitionedCall:output:0)user_fc2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_35522!
tf_op_layer_Mul/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_35672!
tf_op_layer_Sum/PartitionedCall?
IdentityIdentity(tf_op_layer_Sum/PartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4156

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
i
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_3432

inputs
identity?
MaxPoolMaxPoolinputs*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
{
&__inference_doc_fc1_layer_call_fn_4110

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc1_layer_call_and_return_conditional_losses_33702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4141

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
@:::S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
P
4__inference_tf_op_layer_Squeeze_2_layer_call_fn_4212

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_34452
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_4227

inputs
identity?
	Squeeze_3Squeezeinputs*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2
	Squeeze_3f
IdentityIdentitySqueeze_3:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
u
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_4278
inputs_0
inputs_1
identityf
MulMulinputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:?????????@2
Mul[
IdentityIdentityMul:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????@:?????????@:Q M
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1
?
P
4__inference_tf_op_layer_MaxPool_1_layer_call_fn_4192

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_34192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
Z
.__inference_tf_op_layer_Mul_layer_call_fn_4284
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_35522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????@:?????????@:Q M
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????@
"
_user_specified_name
inputs/1
?
Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4161

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_34062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_3788
doc_ids
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsuser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_32432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
H__inference_user_embedding_layer_call_and_return_conditional_losses_3258

inputs
embedding_lookup_3252
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_3252Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/3252*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/3252*+
_output_shapes
:?????????
@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
n
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_3392

inputs
identityf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputsExpandDims_1/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2
ExpandDims_1q
IdentityIdentityExpandDims_1:output:0*
T0*/
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?z
?
?__inference_model_layer_call_and_return_conditional_losses_3984
inputs_0
inputs_1(
$user_embedding_embedding_lookup_3891'
#doc_embedding_embedding_lookup_3897.
*user_fc1_tensordot_readvariableop_resource,
(user_fc1_biasadd_readvariableop_resource-
)doc_fc1_tensordot_readvariableop_resource+
'doc_fc1_biasadd_readvariableop_resource*
&doc_fc2_matmul_readvariableop_resource+
'doc_fc2_biasadd_readvariableop_resource+
'user_fc2_matmul_readvariableop_resource,
(user_fc2_biasadd_readvariableop_resource
identity?}
user_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather$user_embedding_embedding_lookup_3891user_embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@user_embedding/embedding_lookup/3891*+
_output_shapes
:?????????
@*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@user_embedding/embedding_lookup/3891*+
_output_shapes
:?????????
@2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2,
*user_embedding/embedding_lookup/Identity_1{
doc_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
doc_embedding/Cast?
doc_embedding/embedding_lookupResourceGather#doc_embedding_embedding_lookup_3897doc_embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@doc_embedding/embedding_lookup/3897*+
_output_shapes
:?????????
@*
dtype02 
doc_embedding/embedding_lookup?
'doc_embedding/embedding_lookup/IdentityIdentity'doc_embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@doc_embedding/embedding_lookup/3897*+
_output_shapes
:?????????
@2)
'doc_embedding/embedding_lookup/Identity?
)doc_embedding/embedding_lookup/Identity_1Identity0doc_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2+
)doc_embedding/embedding_lookup/Identity_1?
!user_fc1/Tensordot/ReadVariableOpReadVariableOp*user_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!user_fc1/Tensordot/ReadVariableOp|
user_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
user_fc1/Tensordot/axes?
user_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
user_fc1/Tensordot/free?
user_fc1/Tensordot/ShapeShape3user_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
user_fc1/Tensordot/Shape?
 user_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 user_fc1/Tensordot/GatherV2/axis?
user_fc1/Tensordot/GatherV2GatherV2!user_fc1/Tensordot/Shape:output:0 user_fc1/Tensordot/free:output:0)user_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
user_fc1/Tensordot/GatherV2?
"user_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"user_fc1/Tensordot/GatherV2_1/axis?
user_fc1/Tensordot/GatherV2_1GatherV2!user_fc1/Tensordot/Shape:output:0 user_fc1/Tensordot/axes:output:0+user_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
user_fc1/Tensordot/GatherV2_1~
user_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
user_fc1/Tensordot/Const?
user_fc1/Tensordot/ProdProd$user_fc1/Tensordot/GatherV2:output:0!user_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
user_fc1/Tensordot/Prod?
user_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
user_fc1/Tensordot/Const_1?
user_fc1/Tensordot/Prod_1Prod&user_fc1/Tensordot/GatherV2_1:output:0#user_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
user_fc1/Tensordot/Prod_1?
user_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
user_fc1/Tensordot/concat/axis?
user_fc1/Tensordot/concatConcatV2 user_fc1/Tensordot/free:output:0 user_fc1/Tensordot/axes:output:0'user_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/concat?
user_fc1/Tensordot/stackPack user_fc1/Tensordot/Prod:output:0"user_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/stack?
user_fc1/Tensordot/transpose	Transpose3user_embedding/embedding_lookup/Identity_1:output:0"user_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Tensordot/transpose?
user_fc1/Tensordot/ReshapeReshape user_fc1/Tensordot/transpose:y:0!user_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
user_fc1/Tensordot/Reshape?
user_fc1/Tensordot/MatMulMatMul#user_fc1/Tensordot/Reshape:output:0)user_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc1/Tensordot/MatMul?
user_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
user_fc1/Tensordot/Const_2?
 user_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 user_fc1/Tensordot/concat_1/axis?
user_fc1/Tensordot/concat_1ConcatV2$user_fc1/Tensordot/GatherV2:output:0#user_fc1/Tensordot/Const_2:output:0)user_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/concat_1?
user_fc1/TensordotReshape#user_fc1/Tensordot/MatMul:product:0$user_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Tensordot?
user_fc1/BiasAdd/ReadVariableOpReadVariableOp(user_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
user_fc1/BiasAdd/ReadVariableOp?
user_fc1/BiasAddBiasAdduser_fc1/Tensordot:output:0'user_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/BiasAddw
user_fc1/ReluReluuser_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Relu?
 doc_fc1/Tensordot/ReadVariableOpReadVariableOp)doc_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02"
 doc_fc1/Tensordot/ReadVariableOpz
doc_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
doc_fc1/Tensordot/axes?
doc_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
doc_fc1/Tensordot/free?
doc_fc1/Tensordot/ShapeShape2doc_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
doc_fc1/Tensordot/Shape?
doc_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
doc_fc1/Tensordot/GatherV2/axis?
doc_fc1/Tensordot/GatherV2GatherV2 doc_fc1/Tensordot/Shape:output:0doc_fc1/Tensordot/free:output:0(doc_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
doc_fc1/Tensordot/GatherV2?
!doc_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!doc_fc1/Tensordot/GatherV2_1/axis?
doc_fc1/Tensordot/GatherV2_1GatherV2 doc_fc1/Tensordot/Shape:output:0doc_fc1/Tensordot/axes:output:0*doc_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
doc_fc1/Tensordot/GatherV2_1|
doc_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
doc_fc1/Tensordot/Const?
doc_fc1/Tensordot/ProdProd#doc_fc1/Tensordot/GatherV2:output:0 doc_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
doc_fc1/Tensordot/Prod?
doc_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
doc_fc1/Tensordot/Const_1?
doc_fc1/Tensordot/Prod_1Prod%doc_fc1/Tensordot/GatherV2_1:output:0"doc_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
doc_fc1/Tensordot/Prod_1?
doc_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
doc_fc1/Tensordot/concat/axis?
doc_fc1/Tensordot/concatConcatV2doc_fc1/Tensordot/free:output:0doc_fc1/Tensordot/axes:output:0&doc_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/concat?
doc_fc1/Tensordot/stackPackdoc_fc1/Tensordot/Prod:output:0!doc_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/stack?
doc_fc1/Tensordot/transpose	Transpose2doc_embedding/embedding_lookup/Identity_1:output:0!doc_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Tensordot/transpose?
doc_fc1/Tensordot/ReshapeReshapedoc_fc1/Tensordot/transpose:y:0 doc_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
doc_fc1/Tensordot/Reshape?
doc_fc1/Tensordot/MatMulMatMul"doc_fc1/Tensordot/Reshape:output:0(doc_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc1/Tensordot/MatMul?
doc_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
doc_fc1/Tensordot/Const_2?
doc_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
doc_fc1/Tensordot/concat_1/axis?
doc_fc1/Tensordot/concat_1ConcatV2#doc_fc1/Tensordot/GatherV2:output:0"doc_fc1/Tensordot/Const_2:output:0(doc_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/concat_1?
doc_fc1/TensordotReshape"doc_fc1/Tensordot/MatMul:product:0#doc_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Tensordot?
doc_fc1/BiasAdd/ReadVariableOpReadVariableOp'doc_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
doc_fc1/BiasAdd/ReadVariableOp?
doc_fc1/BiasAddBiasAdddoc_fc1/Tensordot:output:0&doc_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/BiasAddt
doc_fc1/ReluReludoc_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Relu?
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)tf_op_layer_ExpandDims_1/ExpandDims_1/dim?
%tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDimsuser_fc1/Relu:activations:02tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2'
%tf_op_layer_ExpandDims_1/ExpandDims_1?
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim?
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsdoc_fc1/Relu:activations:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2#
!tf_op_layer_ExpandDims/ExpandDims?
tf_op_layer_MaxPool_1/MaxPool_1MaxPool.tf_op_layer_ExpandDims_1/ExpandDims_1:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2!
tf_op_layer_MaxPool_1/MaxPool_1?
tf_op_layer_MaxPool/MaxPoolMaxPool*tf_op_layer_ExpandDims/ExpandDims:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2
tf_op_layer_MaxPool/MaxPool?
tf_op_layer_Squeeze_2/Squeeze_2Squeeze(tf_op_layer_MaxPool_1/MaxPool_1:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_2/Squeeze_2?
tf_op_layer_Squeeze/SqueezeSqueeze$tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
tf_op_layer_Squeeze/Squeeze?
tf_op_layer_Squeeze_3/Squeeze_3Squeeze(tf_op_layer_Squeeze_2/Squeeze_2:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_3/Squeeze_3?
tf_op_layer_Squeeze_1/Squeeze_1Squeeze$tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_1/Squeeze_1?
doc_fc2/MatMul/ReadVariableOpReadVariableOp&doc_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
doc_fc2/MatMul/ReadVariableOp?
doc_fc2/MatMulMatMul(tf_op_layer_Squeeze_1/Squeeze_1:output:0%doc_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/MatMul?
doc_fc2/BiasAdd/ReadVariableOpReadVariableOp'doc_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
doc_fc2/BiasAdd/ReadVariableOp?
doc_fc2/BiasAddBiasAdddoc_fc2/MatMul:product:0&doc_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/BiasAddp
doc_fc2/ReluReludoc_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/Relu?
user_fc2/MatMul/ReadVariableOpReadVariableOp'user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
user_fc2/MatMul/ReadVariableOp?
user_fc2/MatMulMatMul(tf_op_layer_Squeeze_3/Squeeze_3:output:0&user_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc2/MatMul?
user_fc2/BiasAdd/ReadVariableOpReadVariableOp(user_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
user_fc2/BiasAdd/ReadVariableOp?
user_fc2/BiasAddBiasAdduser_fc2/MatMul:product:0'user_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc2/BiasAdds
user_fc2/ReluReluuser_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
user_fc2/Relu?
tf_op_layer_Mul/MulMuldoc_fc2/Relu:activations:0user_fc2/Relu:activations:0*
T0*
_cloned(*'
_output_shapes
:?????????@2
tf_op_layer_Mul/Mul?
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices?
tf_op_layer_Sum/SumSumtf_op_layer_Mul/Mul:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Sum/Sump
IdentityIdentitytf_op_layer_Sum/Sum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
:::::::::::Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
e
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_3567

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_4036
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_37292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
B__inference_user_fc2_layer_call_and_return_conditional_losses_4263

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
s
-__inference_user_embedding_layer_call_fn_4070

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_user_embedding_layer_call_and_return_conditional_losses_32582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
?
$__inference_model_layer_call_fn_3685
doc_ids
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsuser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_36622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:QM
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
{
&__inference_doc_fc2_layer_call_fn_4252

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc2_layer_call_and_return_conditional_losses_35032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
?__inference_model_layer_call_and_return_conditional_losses_3729

inputs
inputs_1
user_embedding_3692
doc_embedding_3695
user_fc1_3698
user_fc1_3700
doc_fc1_3703
doc_fc1_3705
doc_fc2_3716
doc_fc2_3718
user_fc2_3721
user_fc2_3723
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_embedding_3692*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_user_embedding_layer_call_and_return_conditional_losses_32582(
&user_embedding/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsdoc_embedding_3695*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_doc_embedding_layer_call_and_return_conditional_losses_32802'
%doc_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_3698user_fc1_3700*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc1_layer_call_and_return_conditional_losses_33232"
 user_fc1/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_3703doc_fc1_3705*
Tin
2*
Tout
2*+
_output_shapes
:?????????
@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc1_layer_call_and_return_conditional_losses_33702!
doc_fc1/StatefulPartitionedCall?
(tf_op_layer_ExpandDims_1/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_33922*
(tf_op_layer_ExpandDims_1/PartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall(doc_fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_34062(
&tf_op_layer_ExpandDims/PartitionedCall?
%tf_op_layer_MaxPool_1/PartitionedCallPartitionedCall1tf_op_layer_ExpandDims_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_34192'
%tf_op_layer_MaxPool_1/PartitionedCall?
#tf_op_layer_MaxPool/PartitionedCallPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_34322%
#tf_op_layer_MaxPool/PartitionedCall?
%tf_op_layer_Squeeze_2/PartitionedCallPartitionedCall.tf_op_layer_MaxPool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_34452'
%tf_op_layer_Squeeze_2/PartitionedCall?
#tf_op_layer_Squeeze/PartitionedCallPartitionedCall,tf_op_layer_MaxPool/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_34582%
#tf_op_layer_Squeeze/PartitionedCall?
%tf_op_layer_Squeeze_3/PartitionedCallPartitionedCall.tf_op_layer_Squeeze_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_34712'
%tf_op_layer_Squeeze_3/PartitionedCall?
%tf_op_layer_Squeeze_1/PartitionedCallPartitionedCall,tf_op_layer_Squeeze/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_34842'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_3716doc_fc2_3718*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_doc_fc2_layer_call_and_return_conditional_losses_35032!
doc_fc2/StatefulPartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_3/PartitionedCall:output:0user_fc2_3721user_fc2_3723*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_user_fc2_layer_call_and_return_conditional_losses_35302"
 user_fc2/StatefulPartitionedCall?
tf_op_layer_Mul/PartitionedCallPartitionedCall(doc_fc2/StatefulPartitionedCall:output:0)user_fc2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_35522!
tf_op_layer_Mul/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall(tf_op_layer_Mul/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_35672!
tf_op_layer_Sum/PartitionedCall?
IdentityIdentity(tf_op_layer_Sum/PartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
::::::::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?z
?
?__inference_model_layer_call_and_return_conditional_losses_3886
inputs_0
inputs_1(
$user_embedding_embedding_lookup_3793'
#doc_embedding_embedding_lookup_3799.
*user_fc1_tensordot_readvariableop_resource,
(user_fc1_biasadd_readvariableop_resource-
)doc_fc1_tensordot_readvariableop_resource+
'doc_fc1_biasadd_readvariableop_resource*
&doc_fc2_matmul_readvariableop_resource+
'doc_fc2_biasadd_readvariableop_resource+
'user_fc2_matmul_readvariableop_resource,
(user_fc2_biasadd_readvariableop_resource
identity?}
user_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather$user_embedding_embedding_lookup_3793user_embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@user_embedding/embedding_lookup/3793*+
_output_shapes
:?????????
@*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@user_embedding/embedding_lookup/3793*+
_output_shapes
:?????????
@2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2,
*user_embedding/embedding_lookup/Identity_1{
doc_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
doc_embedding/Cast?
doc_embedding/embedding_lookupResourceGather#doc_embedding_embedding_lookup_3799doc_embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@doc_embedding/embedding_lookup/3799*+
_output_shapes
:?????????
@*
dtype02 
doc_embedding/embedding_lookup?
'doc_embedding/embedding_lookup/IdentityIdentity'doc_embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@doc_embedding/embedding_lookup/3799*+
_output_shapes
:?????????
@2)
'doc_embedding/embedding_lookup/Identity?
)doc_embedding/embedding_lookup/Identity_1Identity0doc_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2+
)doc_embedding/embedding_lookup/Identity_1?
!user_fc1/Tensordot/ReadVariableOpReadVariableOp*user_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!user_fc1/Tensordot/ReadVariableOp|
user_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
user_fc1/Tensordot/axes?
user_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
user_fc1/Tensordot/free?
user_fc1/Tensordot/ShapeShape3user_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
user_fc1/Tensordot/Shape?
 user_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 user_fc1/Tensordot/GatherV2/axis?
user_fc1/Tensordot/GatherV2GatherV2!user_fc1/Tensordot/Shape:output:0 user_fc1/Tensordot/free:output:0)user_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
user_fc1/Tensordot/GatherV2?
"user_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"user_fc1/Tensordot/GatherV2_1/axis?
user_fc1/Tensordot/GatherV2_1GatherV2!user_fc1/Tensordot/Shape:output:0 user_fc1/Tensordot/axes:output:0+user_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
user_fc1/Tensordot/GatherV2_1~
user_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
user_fc1/Tensordot/Const?
user_fc1/Tensordot/ProdProd$user_fc1/Tensordot/GatherV2:output:0!user_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
user_fc1/Tensordot/Prod?
user_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
user_fc1/Tensordot/Const_1?
user_fc1/Tensordot/Prod_1Prod&user_fc1/Tensordot/GatherV2_1:output:0#user_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
user_fc1/Tensordot/Prod_1?
user_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
user_fc1/Tensordot/concat/axis?
user_fc1/Tensordot/concatConcatV2 user_fc1/Tensordot/free:output:0 user_fc1/Tensordot/axes:output:0'user_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/concat?
user_fc1/Tensordot/stackPack user_fc1/Tensordot/Prod:output:0"user_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/stack?
user_fc1/Tensordot/transpose	Transpose3user_embedding/embedding_lookup/Identity_1:output:0"user_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Tensordot/transpose?
user_fc1/Tensordot/ReshapeReshape user_fc1/Tensordot/transpose:y:0!user_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
user_fc1/Tensordot/Reshape?
user_fc1/Tensordot/MatMulMatMul#user_fc1/Tensordot/Reshape:output:0)user_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc1/Tensordot/MatMul?
user_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
user_fc1/Tensordot/Const_2?
 user_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 user_fc1/Tensordot/concat_1/axis?
user_fc1/Tensordot/concat_1ConcatV2$user_fc1/Tensordot/GatherV2:output:0#user_fc1/Tensordot/Const_2:output:0)user_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
user_fc1/Tensordot/concat_1?
user_fc1/TensordotReshape#user_fc1/Tensordot/MatMul:product:0$user_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Tensordot?
user_fc1/BiasAdd/ReadVariableOpReadVariableOp(user_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
user_fc1/BiasAdd/ReadVariableOp?
user_fc1/BiasAddBiasAdduser_fc1/Tensordot:output:0'user_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/BiasAddw
user_fc1/ReluReluuser_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
user_fc1/Relu?
 doc_fc1/Tensordot/ReadVariableOpReadVariableOp)doc_fc1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02"
 doc_fc1/Tensordot/ReadVariableOpz
doc_fc1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
doc_fc1/Tensordot/axes?
doc_fc1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
doc_fc1/Tensordot/free?
doc_fc1/Tensordot/ShapeShape2doc_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
doc_fc1/Tensordot/Shape?
doc_fc1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
doc_fc1/Tensordot/GatherV2/axis?
doc_fc1/Tensordot/GatherV2GatherV2 doc_fc1/Tensordot/Shape:output:0doc_fc1/Tensordot/free:output:0(doc_fc1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
doc_fc1/Tensordot/GatherV2?
!doc_fc1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!doc_fc1/Tensordot/GatherV2_1/axis?
doc_fc1/Tensordot/GatherV2_1GatherV2 doc_fc1/Tensordot/Shape:output:0doc_fc1/Tensordot/axes:output:0*doc_fc1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
doc_fc1/Tensordot/GatherV2_1|
doc_fc1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
doc_fc1/Tensordot/Const?
doc_fc1/Tensordot/ProdProd#doc_fc1/Tensordot/GatherV2:output:0 doc_fc1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
doc_fc1/Tensordot/Prod?
doc_fc1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
doc_fc1/Tensordot/Const_1?
doc_fc1/Tensordot/Prod_1Prod%doc_fc1/Tensordot/GatherV2_1:output:0"doc_fc1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
doc_fc1/Tensordot/Prod_1?
doc_fc1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
doc_fc1/Tensordot/concat/axis?
doc_fc1/Tensordot/concatConcatV2doc_fc1/Tensordot/free:output:0doc_fc1/Tensordot/axes:output:0&doc_fc1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/concat?
doc_fc1/Tensordot/stackPackdoc_fc1/Tensordot/Prod:output:0!doc_fc1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/stack?
doc_fc1/Tensordot/transpose	Transpose2doc_embedding/embedding_lookup/Identity_1:output:0!doc_fc1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Tensordot/transpose?
doc_fc1/Tensordot/ReshapeReshapedoc_fc1/Tensordot/transpose:y:0 doc_fc1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
doc_fc1/Tensordot/Reshape?
doc_fc1/Tensordot/MatMulMatMul"doc_fc1/Tensordot/Reshape:output:0(doc_fc1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc1/Tensordot/MatMul?
doc_fc1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
doc_fc1/Tensordot/Const_2?
doc_fc1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
doc_fc1/Tensordot/concat_1/axis?
doc_fc1/Tensordot/concat_1ConcatV2#doc_fc1/Tensordot/GatherV2:output:0"doc_fc1/Tensordot/Const_2:output:0(doc_fc1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
doc_fc1/Tensordot/concat_1?
doc_fc1/TensordotReshape"doc_fc1/Tensordot/MatMul:product:0#doc_fc1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Tensordot?
doc_fc1/BiasAdd/ReadVariableOpReadVariableOp'doc_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
doc_fc1/BiasAdd/ReadVariableOp?
doc_fc1/BiasAddBiasAdddoc_fc1/Tensordot:output:0&doc_fc1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/BiasAddt
doc_fc1/ReluReludoc_fc1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????
@2
doc_fc1/Relu?
)tf_op_layer_ExpandDims_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)tf_op_layer_ExpandDims_1/ExpandDims_1/dim?
%tf_op_layer_ExpandDims_1/ExpandDims_1
ExpandDimsuser_fc1/Relu:activations:02tf_op_layer_ExpandDims_1/ExpandDims_1/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2'
%tf_op_layer_ExpandDims_1/ExpandDims_1?
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim?
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsdoc_fc1/Relu:activations:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2#
!tf_op_layer_ExpandDims/ExpandDims?
tf_op_layer_MaxPool_1/MaxPool_1MaxPool.tf_op_layer_ExpandDims_1/ExpandDims_1:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2!
tf_op_layer_MaxPool_1/MaxPool_1?
tf_op_layer_MaxPool/MaxPoolMaxPool*tf_op_layer_ExpandDims/ExpandDims:output:0*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2
tf_op_layer_MaxPool/MaxPool?
tf_op_layer_Squeeze_2/Squeeze_2Squeeze(tf_op_layer_MaxPool_1/MaxPool_1:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_2/Squeeze_2?
tf_op_layer_Squeeze/SqueezeSqueeze$tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
tf_op_layer_Squeeze/Squeeze?
tf_op_layer_Squeeze_3/Squeeze_3Squeeze(tf_op_layer_Squeeze_2/Squeeze_2:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_3/Squeeze_3?
tf_op_layer_Squeeze_1/Squeeze_1Squeeze$tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_1/Squeeze_1?
doc_fc2/MatMul/ReadVariableOpReadVariableOp&doc_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
doc_fc2/MatMul/ReadVariableOp?
doc_fc2/MatMulMatMul(tf_op_layer_Squeeze_1/Squeeze_1:output:0%doc_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/MatMul?
doc_fc2/BiasAdd/ReadVariableOpReadVariableOp'doc_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
doc_fc2/BiasAdd/ReadVariableOp?
doc_fc2/BiasAddBiasAdddoc_fc2/MatMul:product:0&doc_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/BiasAddp
doc_fc2/ReluReludoc_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
doc_fc2/Relu?
user_fc2/MatMul/ReadVariableOpReadVariableOp'user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
user_fc2/MatMul/ReadVariableOp?
user_fc2/MatMulMatMul(tf_op_layer_Squeeze_3/Squeeze_3:output:0&user_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc2/MatMul?
user_fc2/BiasAdd/ReadVariableOpReadVariableOp(user_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
user_fc2/BiasAdd/ReadVariableOp?
user_fc2/BiasAddBiasAdduser_fc2/MatMul:product:0'user_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
user_fc2/BiasAdds
user_fc2/ReluReluuser_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
user_fc2/Relu?
tf_op_layer_Mul/MulMuldoc_fc2/Relu:activations:0user_fc2/Relu:activations:0*
T0*
_cloned(*'
_output_shapes
:?????????@2
tf_op_layer_Mul/Mul?
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices?
tf_op_layer_Sum/SumSumtf_op_layer_Mul/Mul:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Sum/Sump
IdentityIdentitytf_op_layer_Sum/Sum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????
:?????????
:::::::::::Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
P
4__inference_tf_op_layer_Squeeze_3_layer_call_fn_4232

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*X
fSRQ
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_34712
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_doc_embedding_layer_call_and_return_conditional_losses_4046

inputs
embedding_lookup_4040
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_4040Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/4040*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/4040*+
_output_shapes
:?????????
@2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????
@2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????
::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: 
?
i
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4177

inputs
identity?
MaxPoolMaxPoolinputs*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_4187

inputs
identity?
	MaxPool_1MaxPoolinputs*
_cloned(*/
_output_shapes
:?????????@*
ksize

*
paddingSAME*
strides

2
	MaxPool_1n
IdentityIdentityMaxPool_1:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
@:W S
/
_output_shapes
:?????????
@
 
_user_specified_nameinputs
?
s
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_3552

inputs
inputs_1
identityd
MulMulinputsinputs_1*
T0*
_cloned(*'
_output_shapes
:?????????@2
Mul[
IdentityIdentityMul:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????@:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
doc_ids0
serving_default_doc_ids:0?????????

=
user_ids1
serving_default_user_ids:0?????????
C
tf_op_layer_Sum0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer_with_weights-5
layer-15
layer-16
layer-17
	optimizer

signatures
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?~
_tf_keras_model?~{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}, "name": "doc_ids", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}, "name": "user_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "doc_embedding", "inbound_nodes": [[["doc_ids", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "user_embedding", "inbound_nodes": [[["user_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc1", "inbound_nodes": [[["doc_embedding", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc1", "inbound_nodes": [[["user_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["doc_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims_1/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_1", "inbound_nodes": [[["user_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool_1", "op": "MaxPool", "input": ["ExpandDims_1"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}, "name": "tf_op_layer_MaxPool_1", "inbound_nodes": [[["tf_op_layer_ExpandDims_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_2", "op": "Squeeze", "input": ["MaxPool_1"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_2", "inbound_nodes": [[["tf_op_layer_MaxPool_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_3", "op": "Squeeze", "input": ["Squeeze_2"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_3", "inbound_nodes": [[["tf_op_layer_Squeeze_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["doc_fc2/Identity", "user_fc2/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["doc_fc2", 0, 0, {}], ["user_fc2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Mul", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}]]]}], "input_layers": [["doc_ids", 0, 0], ["user_ids", 0, 0]], "output_layers": [["tf_op_layer_Sum", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}, "name": "doc_ids", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}, "name": "user_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "doc_embedding", "inbound_nodes": [[["doc_ids", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "user_embedding", "inbound_nodes": [[["user_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc1", "inbound_nodes": [[["doc_embedding", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc1", "inbound_nodes": [[["user_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["doc_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims_1/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims_1", "inbound_nodes": [[["user_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool_1", "op": "MaxPool", "input": ["ExpandDims_1"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}, "name": "tf_op_layer_MaxPool_1", "inbound_nodes": [[["tf_op_layer_ExpandDims_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_2", "op": "Squeeze", "input": ["MaxPool_1"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_2", "inbound_nodes": [[["tf_op_layer_MaxPool_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_3", "op": "Squeeze", "input": ["Squeeze_2"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_3", "inbound_nodes": [[["tf_op_layer_Squeeze_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["doc_fc2/Identity", "user_fc2/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["doc_fc2", 0, 0, {}], ["user_fc2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Mul", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}]]]}], "input_layers": [["doc_ids", 0, 0], ["user_ids", 0, 0]], "output_layers": [["tf_op_layer_Sum", 0, 0]]}}, "training_config": {"loss": "bi_cross_entropy_loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "doc_ids", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "user_ids", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}}
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "doc_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "stateful": false, "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

embeddings
trainable_variables
 regularization_losses
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "user_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "stateful": false, "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "doc_fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64]}}
?

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "user_fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64]}}
?
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ExpandDims_1", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims_1", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims_1/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPool_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool_1", "op": "MaxPool", "input": ["ExpandDims_1"], "attr": {"T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}}}, "constants": {}}}
?
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze_2", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_2", "op": "Squeeze", "input": ["MaxPool_1"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze_3", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_3", "op": "Squeeze", "input": ["Squeeze_2"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "doc_fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "user_fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["doc_fc2/Identity", "user_fc2/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Mul", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem?m?#m?$m?)m?*m?Om?Pm?Um?Vm?v?v?#v?$v?)v?*v?Ov?Pv?Uv?Vv?"
	optimizer
-
?serving_default"
signature_map
f
0
1
#2
$3
)4
*5
O6
P7
U8
V9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
#2
$3
)4
*5
O6
P7
U8
V9"
trackable_list_wrapper
?
hmetrics

ilayers
jlayer_regularization_losses
trainable_variables
regularization_losses
klayer_metrics
	variables
lnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@2doc_embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
mmetrics

nlayers
olayer_regularization_losses
trainable_variables
regularization_losses
player_metrics
	variables
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@2user_embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
rmetrics

slayers
tlayer_regularization_losses
trainable_variables
 regularization_losses
ulayer_metrics
!	variables
vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@@2doc_fc1/kernel
:@2doc_fc1/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
wmetrics

xlayers
ylayer_regularization_losses
%trainable_variables
&regularization_losses
zlayer_metrics
'	variables
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2user_fc1/kernel
:@2user_fc1/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
|metrics

}layers
~layer_regularization_losses
+trainable_variables
,regularization_losses
layer_metrics
-	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
/trainable_variables
0regularization_losses
?layer_metrics
1	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
3trainable_variables
4regularization_losses
?layer_metrics
5	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
7trainable_variables
8regularization_losses
?layer_metrics
9	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
;trainable_variables
<regularization_losses
?layer_metrics
=	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
?trainable_variables
@regularization_losses
?layer_metrics
A	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
Ctrainable_variables
Dregularization_losses
?layer_metrics
E	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
Gtrainable_variables
Hregularization_losses
?layer_metrics
I	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
Ktrainable_variables
Lregularization_losses
?layer_metrics
M	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@@2doc_fc2/kernel
:@2doc_fc2/bias
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
Qtrainable_variables
Rregularization_losses
?layer_metrics
S	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2user_fc2/kernel
:@2user_fc2/bias
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
Wtrainable_variables
Xregularization_losses
?layer_metrics
Y	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
[trainable_variables
\regularization_losses
?layer_metrics
]	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
 ?layer_regularization_losses
_trainable_variables
`regularization_losses
?layer_metrics
a	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
?0"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-@2Adam/doc_embedding/embeddings/m
0:.@2 Adam/user_embedding/embeddings/m
%:#@@2Adam/doc_fc1/kernel/m
:@2Adam/doc_fc1/bias/m
&:$@@2Adam/user_fc1/kernel/m
 :@2Adam/user_fc1/bias/m
%:#@@2Adam/doc_fc2/kernel/m
:@2Adam/doc_fc2/bias/m
&:$@@2Adam/user_fc2/kernel/m
 :@2Adam/user_fc2/bias/m
/:-@2Adam/doc_embedding/embeddings/v
0:.@2 Adam/user_embedding/embeddings/v
%:#@@2Adam/doc_fc1/kernel/v
:@2Adam/doc_fc1/bias/v
&:$@@2Adam/user_fc1/kernel/v
 :@2Adam/user_fc1/bias/v
%:#@@2Adam/doc_fc2/kernel/v
:@2Adam/doc_fc2/bias/v
&:$@@2Adam/user_fc2/kernel/v
 :@2Adam/user_fc2/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_3886
?__inference_model_layer_call_and_return_conditional_losses_3984
?__inference_model_layer_call_and_return_conditional_losses_3576
?__inference_model_layer_call_and_return_conditional_losses_3617?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_3243?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
!?
doc_ids?????????

"?
user_ids?????????

?2?
$__inference_model_layer_call_fn_3685
$__inference_model_layer_call_fn_3752
$__inference_model_layer_call_fn_4010
$__inference_model_layer_call_fn_4036?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_doc_embedding_layer_call_and_return_conditional_losses_4046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_doc_embedding_layer_call_fn_4053?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_user_embedding_layer_call_and_return_conditional_losses_4063?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_user_embedding_layer_call_fn_4070?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_4101?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_doc_fc1_layer_call_fn_4110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4141?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_user_fc1_layer_call_fn_4150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_4167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_tf_op_layer_ExpandDims_1_layer_call_fn_4172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_tf_op_layer_MaxPool_layer_call_fn_4182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_4187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_MaxPool_1_layer_call_fn_4192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_4197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_tf_op_layer_Squeeze_layer_call_fn_4202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_4207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Squeeze_2_layer_call_fn_4212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_4217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_4222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_4227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Squeeze_3_layer_call_fn_4232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_4243?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_doc_fc2_layer_call_fn_4252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_user_fc2_layer_call_and_return_conditional_losses_4263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_user_fc2_layer_call_fn_4272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_4278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_tf_op_layer_Mul_layer_call_fn_4284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_tf_op_layer_Sum_layer_call_fn_4295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
"__inference_signature_wrapper_3788doc_idsuser_ids?
__inference__wrapped_model_3243?
)*#$OPUVY?V
O?L
J?G
!?
doc_ids?????????

"?
user_ids?????????

? "A?>
<
tf_op_layer_Sum)?&
tf_op_layer_Sum??????????
G__inference_doc_embedding_layer_call_and_return_conditional_losses_4046_/?,
%?"
 ?
inputs?????????

? ")?&
?
0?????????
@
? ?
,__inference_doc_embedding_layer_call_fn_4053R/?,
%?"
 ?
inputs?????????

? "??????????
@?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_4101d#$3?0
)?&
$?!
inputs?????????
@
? ")?&
?
0?????????
@
? ?
&__inference_doc_fc1_layer_call_fn_4110W#$3?0
)?&
$?!
inputs?????????
@
? "??????????
@?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_4243\OP/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? y
&__inference_doc_fc2_layer_call_fn_4252OOP/?,
%?"
 ?
inputs?????????@
? "??????????@?
?__inference_model_layer_call_and_return_conditional_losses_3576?
)*#$OPUVa?^
W?T
J?G
!?
doc_ids?????????

"?
user_ids?????????

p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3617?
)*#$OPUVa?^
W?T
J?G
!?
doc_ids?????????

"?
user_ids?????????

p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3886?
)*#$OPUVb?_
X?U
K?H
"?
inputs/0?????????

"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3984?
)*#$OPUVb?_
X?U
K?H
"?
inputs/0?????????

"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_3685?
)*#$OPUVa?^
W?T
J?G
!?
doc_ids?????????

"?
user_ids?????????

p

 
? "???????????
$__inference_model_layer_call_fn_3752?
)*#$OPUVa?^
W?T
J?G
!?
doc_ids?????????

"?
user_ids?????????

p 

 
? "???????????
$__inference_model_layer_call_fn_4010?
)*#$OPUVb?_
X?U
K?H
"?
inputs/0?????????

"?
inputs/1?????????

p

 
? "???????????
$__inference_model_layer_call_fn_4036?
)*#$OPUVb?_
X?U
K?H
"?
inputs/0?????????

"?
inputs/1?????????

p 

 
? "???????????
"__inference_signature_wrapper_3788?
)*#$OPUVk?h
? 
a?^
,
doc_ids!?
doc_ids?????????

.
user_ids"?
user_ids?????????
"A?>
<
tf_op_layer_Sum)?&
tf_op_layer_Sum??????????
R__inference_tf_op_layer_ExpandDims_1_layer_call_and_return_conditional_losses_4167d3?0
)?&
$?!
inputs?????????
@
? "-?*
#? 
0?????????
@
? ?
7__inference_tf_op_layer_ExpandDims_1_layer_call_fn_4172W3?0
)?&
$?!
inputs?????????
@
? " ??????????
@?
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4156d3?0
)?&
$?!
inputs?????????
@
? "-?*
#? 
0?????????
@
? ?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4161W3?0
)?&
$?!
inputs?????????
@
? " ??????????
@?
O__inference_tf_op_layer_MaxPool_1_layer_call_and_return_conditional_losses_4187h7?4
-?*
(?%
inputs?????????
@
? "-?*
#? 
0?????????@
? ?
4__inference_tf_op_layer_MaxPool_1_layer_call_fn_4192[7?4
-?*
(?%
inputs?????????
@
? " ??????????@?
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4177h7?4
-?*
(?%
inputs?????????
@
? "-?*
#? 
0?????????@
? ?
2__inference_tf_op_layer_MaxPool_layer_call_fn_4182[7?4
-?*
(?%
inputs?????????
@
? " ??????????@?
I__inference_tf_op_layer_Mul_layer_call_and_return_conditional_losses_4278?Z?W
P?M
K?H
"?
inputs/0?????????@
"?
inputs/1?????????@
? "%?"
?
0?????????@
? ?
.__inference_tf_op_layer_Mul_layer_call_fn_4284vZ?W
P?M
K?H
"?
inputs/0?????????@
"?
inputs/1?????????@
? "??????????@?
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_4217\3?0
)?&
$?!
inputs?????????@
? "%?"
?
0?????????@
? ?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_4222O3?0
)?&
$?!
inputs?????????@
? "??????????@?
O__inference_tf_op_layer_Squeeze_2_layer_call_and_return_conditional_losses_4207d7?4
-?*
(?%
inputs?????????@
? ")?&
?
0?????????@
? ?
4__inference_tf_op_layer_Squeeze_2_layer_call_fn_4212W7?4
-?*
(?%
inputs?????????@
? "??????????@?
O__inference_tf_op_layer_Squeeze_3_layer_call_and_return_conditional_losses_4227\3?0
)?&
$?!
inputs?????????@
? "%?"
?
0?????????@
? ?
4__inference_tf_op_layer_Squeeze_3_layer_call_fn_4232O3?0
)?&
$?!
inputs?????????@
? "??????????@?
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_4197d7?4
-?*
(?%
inputs?????????@
? ")?&
?
0?????????@
? ?
2__inference_tf_op_layer_Squeeze_layer_call_fn_4202W7?4
-?*
(?%
inputs?????????@
? "??????????@?
I__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4290X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? }
.__inference_tf_op_layer_Sum_layer_call_fn_4295K/?,
%?"
 ?
inputs?????????@
? "???????????
H__inference_user_embedding_layer_call_and_return_conditional_losses_4063_/?,
%?"
 ?
inputs?????????

? ")?&
?
0?????????
@
? ?
-__inference_user_embedding_layer_call_fn_4070R/?,
%?"
 ?
inputs?????????

? "??????????
@?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4141d)*3?0
)?&
$?!
inputs?????????
@
? ")?&
?
0?????????
@
? ?
'__inference_user_fc1_layer_call_fn_4150W)*3?0
)?&
$?!
inputs?????????
@
? "??????????
@?
B__inference_user_fc2_layer_call_and_return_conditional_losses_4263\UV/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
'__inference_user_fc2_layer_call_fn_4272OUV/?,
%?"
 ?
inputs?????????@
? "??????????@