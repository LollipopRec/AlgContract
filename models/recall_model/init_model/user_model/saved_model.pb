??
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
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
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

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
#
0
1
2
)3
*4
#
0
1
2
)3
*4
 
?
/non_trainable_variables
		variables
0layer_metrics
1metrics

2layers

trainable_variables
regularization_losses
3layer_regularization_losses
 
ig
VARIABLE_VALUEuser_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
4non_trainable_variables
	variables
5metrics
6layer_metrics

7layers
trainable_variables
regularization_losses
8layer_regularization_losses
[Y
VARIABLE_VALUEuser_fc1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEuser_fc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
9non_trainable_variables
	variables
:metrics
;layer_metrics

<layers
trainable_variables
regularization_losses
=layer_regularization_losses
 
 
 
?
>non_trainable_variables
	variables
?metrics
@layer_metrics

Alayers
trainable_variables
regularization_losses
Blayer_regularization_losses
 
 
 
?
Cnon_trainable_variables
	variables
Dmetrics
Elayer_metrics

Flayers
trainable_variables
regularization_losses
Glayer_regularization_losses
 
 
 
?
Hnon_trainable_variables
!	variables
Imetrics
Jlayer_metrics

Klayers
"trainable_variables
#regularization_losses
Llayer_regularization_losses
 
 
 
?
Mnon_trainable_variables
%	variables
Nmetrics
Olayer_metrics

Players
&trainable_variables
'regularization_losses
Qlayer_regularization_losses
[Y
VARIABLE_VALUEuser_fc2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEuser_fc2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
Rnon_trainable_variables
+	variables
Smetrics
Tlayer_metrics

Ulayers
,trainable_variables
-regularization_losses
Vlayer_regularization_losses
 
 
 
8
0
1
2
3
4
5
6
7
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
{
serving_default_user_idsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_user_idsuser_embedding/embeddingsuser_fc1/kerneluser_fc1/biasuser_fc2/kerneluser_fc2/bias*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_4790
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-user_embedding/embeddings/Read/ReadVariableOp#user_fc1/kernel/Read/ReadVariableOp!user_fc1/bias/Read/ReadVariableOp#user_fc2/kernel/Read/ReadVariableOp!user_fc2/bias/Read/ReadVariableOpConst*
Tin
	2*
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
__inference__traced_save_5078
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameuser_embedding/embeddingsuser_fc1/kerneluser_fc1/biasuser_fc2/kerneluser_fc2/bias*
Tin

2*
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
 __inference__traced_restore_5105??
?<
?
?__inference_model_layer_call_and_return_conditional_losses_4888

inputs(
$user_embedding_embedding_lookup_4843.
*user_fc1_tensordot_readvariableop_resource,
(user_fc1_biasadd_readvariableop_resource+
'user_fc2_matmul_readvariableop_resource,
(user_fc2_biasadd_readvariableop_resource
identity?{
user_embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather$user_embedding_embedding_lookup_4843user_embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@user_embedding/embedding_lookup/4843*+
_output_shapes
:?????????
@*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@user_embedding/embedding_lookup/4843*+
_output_shapes
:?????????
@2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2,
*user_embedding/embedding_lookup/Identity_1?
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
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim?
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsuser_fc1/Relu:activations:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2#
!tf_op_layer_ExpandDims/ExpandDims?
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
tf_op_layer_Squeeze/SqueezeSqueeze$tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
tf_op_layer_Squeeze/Squeeze?
tf_op_layer_Squeeze_1/Squeeze_1Squeeze$tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_1/Squeeze_1?
user_fc2/MatMul/ReadVariableOpReadVariableOp'user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
user_fc2/MatMul/ReadVariableOp?
user_fc2/MatMulMatMul(tf_op_layer_Squeeze_1/Squeeze_1:output:0&user_fc2/MatMul/ReadVariableOp:value:0*
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
user_fc2/Reluo
IdentityIdentityuser_fc2/Relu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
::::::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: 
?
?
?__inference_model_layer_call_and_return_conditional_losses_4700
user_ids
user_embedding_4682
user_fc1_4685
user_fc1_4687
user_fc2_4694
user_fc2_4696
identity??&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_idsuser_embedding_4682*
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_45392(
&user_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_4685user_fc1_4687*
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
B__inference_user_fc1_layer_call_and_return_conditional_losses_45822"
 user_fc1/StatefulPartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_46042(
&tf_op_layer_ExpandDims/PartitionedCall?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_46172%
#tf_op_layer_MaxPool/PartitionedCall?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_46302%
#tf_op_layer_Squeeze/PartitionedCall?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_46432'
%tf_op_layer_Squeeze_1/PartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0user_fc2_4694user_fc2_4696*
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
B__inference_user_fc2_layer_call_and_return_conditional_losses_46622"
 user_fc2/StatefulPartitionedCall?
IdentityIdentity)user_fc2/StatefulPartitionedCall:output:0'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
i
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4617

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
?
P
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5016

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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_46432
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_4539

inputs
embedding_lookup_4533
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_4533Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/4533*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/4533*+
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
?
?__inference_model_layer_call_and_return_conditional_losses_4760

inputs
user_embedding_4742
user_fc1_4745
user_fc1_4747
user_fc2_4754
user_fc2_4756
identity??&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_4742*
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_45392(
&user_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_4745user_fc1_4747*
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
B__inference_user_fc1_layer_call_and_return_conditional_losses_45822"
 user_fc1/StatefulPartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_46042(
&tf_op_layer_ExpandDims/PartitionedCall?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_46172%
#tf_op_layer_MaxPool/PartitionedCall?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_46302%
#tf_op_layer_Squeeze/PartitionedCall?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_46432'
%tf_op_layer_Squeeze_1/PartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0user_fc2_4754user_fc2_4756*
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
B__inference_user_fc2_layer_call_and_return_conditional_losses_46622"
 user_fc2/StatefulPartitionedCall?
IdentityIdentity)user_fc2/StatefulPartitionedCall:output:0'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: 
?
?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4966

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
?
?
?__inference_model_layer_call_and_return_conditional_losses_4679
user_ids
user_embedding_4548
user_fc1_4593
user_fc1_4595
user_fc2_4673
user_fc2_4675
identity??&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCalluser_idsuser_embedding_4548*
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_45392(
&user_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_4593user_fc1_4595*
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
B__inference_user_fc1_layer_call_and_return_conditional_losses_45822"
 user_fc1/StatefulPartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_46042(
&tf_op_layer_ExpandDims/PartitionedCall?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_46172%
#tf_op_layer_MaxPool/PartitionedCall?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_46302%
#tf_op_layer_Squeeze/PartitionedCall?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_46432'
%tf_op_layer_Squeeze_1/PartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0user_fc2_4673user_fc2_4675*
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
B__inference_user_fc2_layer_call_and_return_conditional_losses_46622"
 user_fc2/StatefulPartitionedCall?
IdentityIdentity)user_fc2/StatefulPartitionedCall:output:0'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
?
$__inference_model_layer_call_fn_4918

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: 
?
|
'__inference_user_fc2_layer_call_fn_5036

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
B__inference_user_fc2_layer_call_and_return_conditional_losses_46622
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
?!
?
__inference__traced_save_5078
file_prefix8
4savev2_user_embedding_embeddings_read_readvariableop.
*savev2_user_fc1_kernel_read_readvariableop,
(savev2_user_fc1_bias_read_readvariableop.
*savev2_user_fc2_kernel_read_readvariableop,
(savev2_user_fc2_bias_read_readvariableop
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
value3B1 B+_temp_7c5318e7ae1240a5b7e78bb79b33d6b5/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_user_embedding_embeddings_read_readvariableop*savev2_user_fc1_kernel_read_readvariableop(savev2_user_fc1_bias_read_readvariableop*savev2_user_fc2_kernel_read_readvariableop(savev2_user_fc2_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes	
22
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

identity_1Identity_1:output:0*A
_input_shapes0
.: :@:@@:@:@@:@: 2(
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

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
?
i
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5001

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
?
?
?__inference_model_layer_call_and_return_conditional_losses_4724

inputs
user_embedding_4706
user_fc1_4709
user_fc1_4711
user_fc2_4718
user_fc2_4720
identity??&user_embedding/StatefulPartitionedCall? user_fc1/StatefulPartitionedCall? user_fc2/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsuser_embedding_4706*
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_45392(
&user_embedding/StatefulPartitionedCall?
 user_fc1/StatefulPartitionedCallStatefulPartitionedCall/user_embedding/StatefulPartitionedCall:output:0user_fc1_4709user_fc1_4711*
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
B__inference_user_fc1_layer_call_and_return_conditional_losses_45822"
 user_fc1/StatefulPartitionedCall?
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCall)user_fc1/StatefulPartitionedCall:output:0*
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_46042(
&tf_op_layer_ExpandDims/PartitionedCall?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_46172%
#tf_op_layer_MaxPool/PartitionedCall?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_46302%
#tf_op_layer_Squeeze/PartitionedCall?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_46432'
%tf_op_layer_Squeeze_1/PartitionedCall?
 user_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0user_fc2_4718user_fc2_4720*
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
B__inference_user_fc2_layer_call_and_return_conditional_losses_46622"
 user_fc2/StatefulPartitionedCall?
IdentityIdentity)user_fc2/StatefulPartitionedCall:output:0'^user_embedding/StatefulPartitionedCall!^user_fc1/StatefulPartitionedCall!^user_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2D
 user_fc1/StatefulPartitionedCall user_fc1/StatefulPartitionedCall2D
 user_fc2/StatefulPartitionedCall user_fc2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: 
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5011

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
?
N
2__inference_tf_op_layer_Squeeze_layer_call_fn_5006

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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_46302
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4991

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
?
N
2__inference_tf_op_layer_MaxPool_layer_call_fn_4996

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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_46172
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
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_4643

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
B__inference_user_fc2_layer_call_and_return_conditional_losses_4662

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
?
|
'__inference_user_fc1_layer_call_fn_4975

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
B__inference_user_fc1_layer_call_and_return_conditional_losses_45822
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
i
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_4630

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
?B
?
__inference__wrapped_model_4525
user_ids.
*model_user_embedding_embedding_lookup_44804
0model_user_fc1_tensordot_readvariableop_resource2
.model_user_fc1_biasadd_readvariableop_resource1
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
%model/user_embedding/embedding_lookupResourceGather*model_user_embedding_embedding_lookup_4480model/user_embedding/Cast:y:0*
Tindices0*=
_class3
1/loc:@model/user_embedding/embedding_lookup/4480*+
_output_shapes
:?????????
@*
dtype02'
%model/user_embedding/embedding_lookup?
.model/user_embedding/embedding_lookup/IdentityIdentity.model/user_embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model/user_embedding/embedding_lookup/4480*+
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
+model/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model/tf_op_layer_ExpandDims/ExpandDims/dim?
'model/tf_op_layer_ExpandDims/ExpandDims
ExpandDims!model/user_fc1/Relu:activations:04model/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2)
'model/tf_op_layer_ExpandDims/ExpandDims?
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
!model/tf_op_layer_Squeeze/SqueezeSqueeze*model/tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2#
!model/tf_op_layer_Squeeze/Squeeze?
%model/tf_op_layer_Squeeze_1/Squeeze_1Squeeze*model/tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2'
%model/tf_op_layer_Squeeze_1/Squeeze_1?
$model/user_fc2/MatMul/ReadVariableOpReadVariableOp-model_user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02&
$model/user_fc2/MatMul/ReadVariableOp?
model/user_fc2/MatMulMatMul.model/tf_op_layer_Squeeze_1/Squeeze_1:output:0,model/user_fc2/MatMul/ReadVariableOp:value:0*
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
model/user_fc2/Reluu
IdentityIdentity!model/user_fc2/Relu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
::::::Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
?
 __inference__traced_restore_5105
file_prefix.
*assignvariableop_user_embedding_embeddings&
"assignvariableop_1_user_fc1_kernel$
 assignvariableop_2_user_fc1_bias&
"assignvariableop_3_user_fc2_kernel$
 assignvariableop_4_user_fc2_bias

identity_6??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_user_embedding_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_user_fc1_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_user_fc1_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_user_fc2_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_user_fc2_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4?
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
NoOp?

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5?

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42
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
: 
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4604

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
?
?
H__inference_user_embedding_layer_call_and_return_conditional_losses_4928

inputs
embedding_lookup_4922
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_4922Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/4922*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/4922*+
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
?<
?
?__inference_model_layer_call_and_return_conditional_losses_4839

inputs(
$user_embedding_embedding_lookup_4794.
*user_fc1_tensordot_readvariableop_resource,
(user_fc1_biasadd_readvariableop_resource+
'user_fc2_matmul_readvariableop_resource,
(user_fc2_biasadd_readvariableop_resource
identity?{
user_embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather$user_embedding_embedding_lookup_4794user_embedding/Cast:y:0*
Tindices0*7
_class-
+)loc:@user_embedding/embedding_lookup/4794*+
_output_shapes
:?????????
@*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@user_embedding/embedding_lookup/4794*+
_output_shapes
:?????????
@2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
@2,
*user_embedding/embedding_lookup/Identity_1?
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
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dim?
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsuser_fc1/Relu:activations:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*/
_output_shapes
:?????????
@2#
!tf_op_layer_ExpandDims/ExpandDims?
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
tf_op_layer_Squeeze/SqueezeSqueeze$tf_op_layer_MaxPool/MaxPool:output:0*
T0*
_cloned(*+
_output_shapes
:?????????@*
squeeze_dims
2
tf_op_layer_Squeeze/Squeeze?
tf_op_layer_Squeeze_1/Squeeze_1Squeeze$tf_op_layer_Squeeze/Squeeze:output:0*
T0*
_cloned(*'
_output_shapes
:?????????@*
squeeze_dims
2!
tf_op_layer_Squeeze_1/Squeeze_1?
user_fc2/MatMul/ReadVariableOpReadVariableOp'user_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
user_fc2/MatMul/ReadVariableOp?
user_fc2/MatMulMatMul(tf_op_layer_Squeeze_1/Squeeze_1:output:0&user_fc2/MatMul/ReadVariableOp:value:0*
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
user_fc2/Reluo
IdentityIdentityuser_fc2/Relu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
::::::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: 
?
?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4582

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
?
?
$__inference_model_layer_call_fn_4737
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4981

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
?
?
B__inference_user_fc2_layer_call_and_return_conditional_losses_5027

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
?
?
$__inference_model_layer_call_fn_4773
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
s
-__inference_user_embedding_layer_call_fn_4935

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
H__inference_user_embedding_layer_call_and_return_conditional_losses_45392
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
"__inference_signature_wrapper_4790
user_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalluser_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_45252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
user_ids:
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
: 
?
Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4986

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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_46042
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
?
?
$__inference_model_layer_call_fn_4903

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:?????????@*'
_read_only_resource_inputs	
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_47242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:
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
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
user_ids1
serving_default_user_ids:0?????????
<
user_fc20
StatefulPartitionedCall:0?????????@tensorflow/serving/predict:??
?<
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
W_default_save_signature
*X&call_and_return_all_conditional_losses
Y__call__"?9
_tf_keras_model?9{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}, "name": "user_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "user_embedding", "inbound_nodes": [[["user_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc1", "inbound_nodes": [[["user_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["user_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"T": {"type": "DT_FLOAT"}, "squeeze_dims": {"list": {"i": ["1"]}}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}], "input_layers": [["user_ids", 0, 0]], "output_layers": [["user_fc2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}, "name": "user_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "user_embedding", "inbound_nodes": [[["user_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc1", "inbound_nodes": [[["user_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["user_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"T": {"type": "DT_FLOAT"}, "squeeze_dims": {"list": {"i": ["1"]}}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "user_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}], "input_layers": [["user_ids", 0, 0]], "output_layers": [["user_fc2", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "user_ids", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_ids"}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "user_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "stateful": false, "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "user_fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "user_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["user_fc1/Identity", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?
	variables
trainable_variables
regularization_losses
 	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}, "ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"T": {"type": "DT_FLOAT"}, "squeeze_dims": {"list": {"i": ["1"]}}}}, "constants": {}}}
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "user_fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "user_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
C
0
1
2
)3
*4"
trackable_list_wrapper
C
0
1
2
)3
*4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
/non_trainable_variables
		variables
0layer_metrics
1metrics

2layers

trainable_variables
regularization_losses
3layer_regularization_losses
Y__call__
W_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
+:)@2user_embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4non_trainable_variables
	variables
5metrics
6layer_metrics

7layers
trainable_variables
regularization_losses
8layer_regularization_losses
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
!:@@2user_fc1/kernel
:@2user_fc1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables
	variables
:metrics
;layer_metrics

<layers
trainable_variables
regularization_losses
=layer_regularization_losses
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables
	variables
?metrics
@layer_metrics

Alayers
trainable_variables
regularization_losses
Blayer_regularization_losses
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables
	variables
Dmetrics
Elayer_metrics

Flayers
trainable_variables
regularization_losses
Glayer_regularization_losses
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables
!	variables
Imetrics
Jlayer_metrics

Klayers
"trainable_variables
#regularization_losses
Llayer_regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables
%	variables
Nmetrics
Olayer_metrics

Players
&trainable_variables
'regularization_losses
Qlayer_regularization_losses
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
!:@@2user_fc2/kernel
:@2user_fc2/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables
+	variables
Smetrics
Tlayer_metrics

Ulayers
,trainable_variables
-regularization_losses
Vlayer_regularization_losses
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
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
?2?
__inference__wrapped_model_4525?
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
annotations? *'?$
"?
user_ids?????????

?2?
?__inference_model_layer_call_and_return_conditional_losses_4888
?__inference_model_layer_call_and_return_conditional_losses_4679
?__inference_model_layer_call_and_return_conditional_losses_4839
?__inference_model_layer_call_and_return_conditional_losses_4700?
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
$__inference_model_layer_call_fn_4773
$__inference_model_layer_call_fn_4918
$__inference_model_layer_call_fn_4903
$__inference_model_layer_call_fn_4737?
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
H__inference_user_embedding_layer_call_and_return_conditional_losses_4928?
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
-__inference_user_embedding_layer_call_fn_4935?
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
B__inference_user_fc1_layer_call_and_return_conditional_losses_4966?
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
'__inference_user_fc1_layer_call_fn_4975?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4981?
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
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4986?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4991?
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
2__inference_tf_op_layer_MaxPool_layer_call_fn_4996?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5001?
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
2__inference_tf_op_layer_Squeeze_layer_call_fn_5006?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5011?
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
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5016?
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
B__inference_user_fc2_layer_call_and_return_conditional_losses_5027?
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
'__inference_user_fc2_layer_call_fn_5036?
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
2B0
"__inference_signature_wrapper_4790user_ids?
__inference__wrapped_model_4525o)*1?.
'?$
"?
user_ids?????????

? "3?0
.
user_fc2"?
user_fc2?????????@?
?__inference_model_layer_call_and_return_conditional_losses_4679i)*9?6
/?,
"?
user_ids?????????

p

 
? "%?"
?
0?????????@
? ?
?__inference_model_layer_call_and_return_conditional_losses_4700i)*9?6
/?,
"?
user_ids?????????

p 

 
? "%?"
?
0?????????@
? ?
?__inference_model_layer_call_and_return_conditional_losses_4839g)*7?4
-?*
 ?
inputs?????????

p

 
? "%?"
?
0?????????@
? ?
?__inference_model_layer_call_and_return_conditional_losses_4888g)*7?4
-?*
 ?
inputs?????????

p 

 
? "%?"
?
0?????????@
? ?
$__inference_model_layer_call_fn_4737\)*9?6
/?,
"?
user_ids?????????

p

 
? "??????????@?
$__inference_model_layer_call_fn_4773\)*9?6
/?,
"?
user_ids?????????

p 

 
? "??????????@?
$__inference_model_layer_call_fn_4903Z)*7?4
-?*
 ?
inputs?????????

p

 
? "??????????@?
$__inference_model_layer_call_fn_4918Z)*7?4
-?*
 ?
inputs?????????

p 

 
? "??????????@?
"__inference_signature_wrapper_4790{)*=?:
? 
3?0
.
user_ids"?
user_ids?????????
"3?0
.
user_fc2"?
user_fc2?????????@?
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_4981d3?0
)?&
$?!
inputs?????????
@
? "-?*
#? 
0?????????
@
? ?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_4986W3?0
)?&
$?!
inputs?????????
@
? " ??????????
@?
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_4991h7?4
-?*
(?%
inputs?????????
@
? "-?*
#? 
0?????????@
? ?
2__inference_tf_op_layer_MaxPool_layer_call_fn_4996[7?4
-?*
(?%
inputs?????????
@
? " ??????????@?
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5011\3?0
)?&
$?!
inputs?????????@
? "%?"
?
0?????????@
? ?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5016O3?0
)?&
$?!
inputs?????????@
? "??????????@?
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5001d7?4
-?*
(?%
inputs?????????@
? ")?&
?
0?????????@
? ?
2__inference_tf_op_layer_Squeeze_layer_call_fn_5006W7?4
-?*
(?%
inputs?????????@
? "??????????@?
H__inference_user_embedding_layer_call_and_return_conditional_losses_4928_/?,
%?"
 ?
inputs?????????

? ")?&
?
0?????????
@
? ?
-__inference_user_embedding_layer_call_fn_4935R/?,
%?"
 ?
inputs?????????

? "??????????
@?
B__inference_user_fc1_layer_call_and_return_conditional_losses_4966d3?0
)?&
$?!
inputs?????????
@
? ")?&
?
0?????????
@
? ?
'__inference_user_fc1_layer_call_fn_4975W3?0
)?&
$?!
inputs?????????
@
? "??????????
@?
B__inference_user_fc2_layer_call_and_return_conditional_losses_5027\)*/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
'__inference_user_fc2_layer_call_fn_5036O)*/?,
%?"
 ?
inputs?????????@
? "??????????@