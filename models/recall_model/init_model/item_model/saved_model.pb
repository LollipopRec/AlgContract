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
hf
VARIABLE_VALUEdoc_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdoc_fc1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdoc_fc1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ZX
VARIABLE_VALUEdoc_fc2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdoc_fc2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
z
serving_default_doc_idsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_doc_idsdoc_embedding/embeddingsdoc_fc1/kerneldoc_fc1/biasdoc_fc2/kerneldoc_fc2/bias*
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
"__inference_signature_wrapper_5562
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,doc_embedding/embeddings/Read/ReadVariableOp"doc_fc1/kernel/Read/ReadVariableOp doc_fc1/bias/Read/ReadVariableOp"doc_fc2/kernel/Read/ReadVariableOp doc_fc2/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_5850
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedoc_embedding/embeddingsdoc_fc1/kerneldoc_fc1/biasdoc_fc2/kerneldoc_fc2/bias*
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
 __inference__traced_restore_5877??
?
?
$__inference_model_layer_call_fn_5690

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
?__inference_model_layer_call_and_return_conditional_losses_55322
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
?;
?
?__inference_model_layer_call_and_return_conditional_losses_5660

inputs'
#doc_embedding_embedding_lookup_5615-
)doc_fc1_tensordot_readvariableop_resource+
'doc_fc1_biasadd_readvariableop_resource*
&doc_fc2_matmul_readvariableop_resource+
'doc_fc2_biasadd_readvariableop_resource
identity?y
doc_embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
doc_embedding/Cast?
doc_embedding/embedding_lookupResourceGather#doc_embedding_embedding_lookup_5615doc_embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@doc_embedding/embedding_lookup/5615*+
_output_shapes
:?????????
@*
dtype02 
doc_embedding/embedding_lookup?
'doc_embedding/embedding_lookup/IdentityIdentity'doc_embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@doc_embedding/embedding_lookup/5615*+
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
doc_fc2/Relun
IdentityIdentitydoc_fc2/Relu:activations:0*
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
?
i
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_5389

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
?
?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_5434

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
?;
?
?__inference_model_layer_call_and_return_conditional_losses_5611

inputs'
#doc_embedding_embedding_lookup_5566-
)doc_fc1_tensordot_readvariableop_resource+
'doc_fc1_biasadd_readvariableop_resource*
&doc_fc2_matmul_readvariableop_resource+
'doc_fc2_biasadd_readvariableop_resource
identity?y
doc_embedding/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
doc_embedding/Cast?
doc_embedding/embedding_lookupResourceGather#doc_embedding_embedding_lookup_5566doc_embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@doc_embedding/embedding_lookup/5566*+
_output_shapes
:?????????
@*
dtype02 
doc_embedding/embedding_lookup?
'doc_embedding/embedding_lookup/IdentityIdentity'doc_embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@doc_embedding/embedding_lookup/5566*+
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
doc_fc2/Relun
IdentityIdentitydoc_fc2/Relu:activations:0*
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
?__inference_model_layer_call_and_return_conditional_losses_5496

inputs
doc_embedding_5478
doc_fc1_5481
doc_fc1_5483
doc_fc2_5490
doc_fc2_5492
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsdoc_embedding_5478*
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_53112'
%doc_embedding/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_5481doc_fc1_5483*
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_53542!
doc_fc1/StatefulPartitionedCall?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_53762(
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_53892%
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_54022%
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_54152'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_5490doc_fc2_5492*
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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_54342!
doc_fc2/StatefulPartitionedCall?
IdentityIdentity(doc_fc2/StatefulPartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall:O K
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
?
N
2__inference_tf_op_layer_MaxPool_layer_call_fn_5768

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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_53892
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
$__inference_model_layer_call_fn_5509
doc_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
?__inference_model_layer_call_and_return_conditional_losses_54962
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
"__inference_signature_wrapper_5562
doc_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
__inference__wrapped_model_52972
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5376

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
?
r
,__inference_doc_embedding_layer_call_fn_5707

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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_53112
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
?
N
2__inference_tf_op_layer_Squeeze_layer_call_fn_5778

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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_54022
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5783

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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_5799

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
{
&__inference_doc_fc1_layer_call_fn_5747

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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_53542
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_5738

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
?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_5354

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
?
i
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5773

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
?__inference_model_layer_call_and_return_conditional_losses_5532

inputs
doc_embedding_5514
doc_fc1_5517
doc_fc1_5519
doc_fc2_5526
doc_fc2_5528
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsdoc_embedding_5514*
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_53112'
%doc_embedding/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_5517doc_fc1_5519*
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_53542!
doc_fc1/StatefulPartitionedCall?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_53762(
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_53892%
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_54022%
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_54152'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_5526doc_fc2_5528*
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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_54342!
doc_fc2/StatefulPartitionedCall?
IdentityIdentity(doc_fc2/StatefulPartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall:O K
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
?
i
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_5763

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
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5788

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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_54152
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
?!
?
__inference__traced_save_5850
file_prefix7
3savev2_doc_embedding_embeddings_read_readvariableop-
)savev2_doc_fc1_kernel_read_readvariableop+
'savev2_doc_fc1_bias_read_readvariableop-
)savev2_doc_fc2_kernel_read_readvariableop+
'savev2_doc_fc2_bias_read_readvariableop
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
value3B1 B+_temp_403ce49e43624c0488f399a2ba2e1edf/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_doc_embedding_embeddings_read_readvariableop)savev2_doc_fc1_kernel_read_readvariableop'savev2_doc_fc1_bias_read_readvariableop)savev2_doc_fc2_kernel_read_readvariableop'savev2_doc_fc2_bias_read_readvariableop"/device:CPU:0*
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
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5753

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
$__inference_model_layer_call_fn_5545
doc_ids
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldoc_idsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
?__inference_model_layer_call_and_return_conditional_losses_55322
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
$__inference_model_layer_call_fn_5675

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
?__inference_model_layer_call_and_return_conditional_losses_54962
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
?A
?
__inference__wrapped_model_5297
doc_ids-
)model_doc_embedding_embedding_lookup_52523
/model_doc_fc1_tensordot_readvariableop_resource1
-model_doc_fc1_biasadd_readvariableop_resource0
,model_doc_fc2_matmul_readvariableop_resource1
-model_doc_fc2_biasadd_readvariableop_resource
identity??
model/doc_embedding/CastCastdoc_ids*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
model/doc_embedding/Cast?
$model/doc_embedding/embedding_lookupResourceGather)model_doc_embedding_embedding_lookup_5252model/doc_embedding/Cast:y:0*
Tindices0*<
_class2
0.loc:@model/doc_embedding/embedding_lookup/5252*+
_output_shapes
:?????????
@*
dtype02&
$model/doc_embedding/embedding_lookup?
-model/doc_embedding/embedding_lookup/IdentityIdentity-model/doc_embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@model/doc_embedding/embedding_lookup/5252*+
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
model/doc_fc2/Relut
IdentityIdentity model/doc_fc2/Relu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
::::::P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_5700

inputs
embedding_lookup_5694
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_5694Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/5694*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/5694*+
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
?
?
G__inference_doc_embedding_layer_call_and_return_conditional_losses_5311

inputs
embedding_lookup_5305
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????
2
Cast?
embedding_lookupResourceGatherembedding_lookup_5305Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/5305*+
_output_shapes
:?????????
@*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/5305*+
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5402

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
?
k
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5415

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
?
?
 __inference__traced_restore_5877
file_prefix-
)assignvariableop_doc_embedding_embeddings%
!assignvariableop_1_doc_fc1_kernel#
assignvariableop_2_doc_fc1_bias%
!assignvariableop_3_doc_fc2_kernel#
assignvariableop_4_doc_fc2_bias

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
AssignVariableOp_1AssignVariableOp!assignvariableop_1_doc_fc1_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_doc_fc1_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_doc_fc2_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_doc_fc2_biasIdentity_4:output:0*
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
?
?
?__inference_model_layer_call_and_return_conditional_losses_5451
doc_ids
doc_embedding_5320
doc_fc1_5365
doc_fc1_5367
doc_fc2_5445
doc_fc2_5447
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCalldoc_idsdoc_embedding_5320*
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_53112'
%doc_embedding/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_5365doc_fc1_5367*
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_53542!
doc_fc1/StatefulPartitionedCall?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_53762(
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_53892%
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_54022%
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_54152'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_5445doc_fc2_5447*
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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_54342!
doc_fc2/StatefulPartitionedCall?
IdentityIdentity(doc_fc2/StatefulPartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
{
&__inference_doc_fc2_layer_call_fn_5808

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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_54342
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
?
?
?__inference_model_layer_call_and_return_conditional_losses_5472
doc_ids
doc_embedding_5454
doc_fc1_5457
doc_fc1_5459
doc_fc2_5466
doc_fc2_5468
identity??%doc_embedding/StatefulPartitionedCall?doc_fc1/StatefulPartitionedCall?doc_fc2/StatefulPartitionedCall?
%doc_embedding/StatefulPartitionedCallStatefulPartitionedCalldoc_idsdoc_embedding_5454*
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_53112'
%doc_embedding/StatefulPartitionedCall?
doc_fc1/StatefulPartitionedCallStatefulPartitionedCall.doc_embedding/StatefulPartitionedCall:output:0doc_fc1_5457doc_fc1_5459*
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_53542!
doc_fc1/StatefulPartitionedCall?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_53762(
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_53892%
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_54022%
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_54152'
%tf_op_layer_Squeeze_1/PartitionedCall?
doc_fc2/StatefulPartitionedCallStatefulPartitionedCall.tf_op_layer_Squeeze_1/PartitionedCall:output:0doc_fc2_5466doc_fc2_5468*
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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_54342!
doc_fc2/StatefulPartitionedCall?
IdentityIdentity(doc_fc2/StatefulPartitionedCall:output:0&^doc_embedding/StatefulPartitionedCall ^doc_fc1/StatefulPartitionedCall ^doc_fc2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????
:::::2N
%doc_embedding/StatefulPartitionedCall%doc_embedding/StatefulPartitionedCall2B
doc_fc1/StatefulPartitionedCalldoc_fc1/StatefulPartitionedCall2B
doc_fc2/StatefulPartitionedCalldoc_fc2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????

!
_user_specified_name	doc_ids:
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
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5758

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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_53762
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
;
doc_fc20
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
_tf_keras_model?9{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}, "name": "doc_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "doc_embedding", "inbound_nodes": [[["doc_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc1", "inbound_nodes": [[["doc_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["doc_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}], "input_layers": [["doc_ids", 0, 0]], "output_layers": [["doc_fc2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}, "name": "doc_ids", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "name": "doc_embedding", "inbound_nodes": [[["doc_ids", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc1", "inbound_nodes": [[["doc_embedding", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["doc_fc1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}}}, "constants": {}}, "name": "tf_op_layer_MaxPool", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze", "inbound_nodes": [[["tf_op_layer_MaxPool", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Squeeze_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze_1", "op": "Squeeze", "input": ["Squeeze"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Squeeze_1", "inbound_nodes": [[["tf_op_layer_Squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "doc_fc2", "inbound_nodes": [[["tf_op_layer_Squeeze_1", 0, 0, {}]]]}], "input_layers": [["doc_ids", 0, 0]], "output_layers": [["doc_fc2", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "doc_ids", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "doc_ids"}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "doc_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "stateful": false, "config": {"name": "doc_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "input_dim": 20, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "doc_fc1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "doc_fc1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 64]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["doc_fc1/Identity", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": 1}}}
?
	variables
trainable_variables
regularization_losses
 	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MaxPool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "MaxPool", "trainable": true, "dtype": "float32", "node_def": {"name": "MaxPool", "op": "MaxPool", "input": ["ExpandDims"], "attr": {"ksize": {"list": {"i": ["1", "1", "10", "1"]}}, "padding": {"s": "U0FNRQ=="}, "T": {"type": "DT_FLOAT"}, "data_format": {"s": "TkhXQw=="}, "strides": {"list": {"i": ["1", "1", "10", "1"]}}}}, "constants": {}}}
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Squeeze", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Squeeze", "trainable": true, "dtype": "float32", "node_def": {"name": "Squeeze", "op": "Squeeze", "input": ["MaxPool"], "attr": {"squeeze_dims": {"list": {"i": ["1"]}}, "T": {"type": "DT_FLOAT"}}}, "constants": {}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "doc_fc2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "doc_fc2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
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
*:(@2doc_embedding/embeddings
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
 :@@2doc_fc1/kernel
:@2doc_fc1/bias
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
 :@@2doc_fc2/kernel
:@2doc_fc2/bias
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
__inference__wrapped_model_5297?
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
annotations? *&?#
!?
doc_ids?????????

?2?
?__inference_model_layer_call_and_return_conditional_losses_5611
?__inference_model_layer_call_and_return_conditional_losses_5660
?__inference_model_layer_call_and_return_conditional_losses_5472
?__inference_model_layer_call_and_return_conditional_losses_5451?
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
$__inference_model_layer_call_fn_5675
$__inference_model_layer_call_fn_5690
$__inference_model_layer_call_fn_5509
$__inference_model_layer_call_fn_5545?
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
G__inference_doc_embedding_layer_call_and_return_conditional_losses_5700?
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
,__inference_doc_embedding_layer_call_fn_5707?
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
A__inference_doc_fc1_layer_call_and_return_conditional_losses_5738?
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
&__inference_doc_fc1_layer_call_fn_5747?
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5753?
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
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5758?
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
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_5763?
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
2__inference_tf_op_layer_MaxPool_layer_call_fn_5768?
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
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5773?
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
2__inference_tf_op_layer_Squeeze_layer_call_fn_5778?
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
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5783?
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
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5788?
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
A__inference_doc_fc2_layer_call_and_return_conditional_losses_5799?
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
&__inference_doc_fc2_layer_call_fn_5808?
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
1B/
"__inference_signature_wrapper_5562doc_ids?
__inference__wrapped_model_5297l)*0?-
&?#
!?
doc_ids?????????

? "1?.
,
doc_fc2!?
doc_fc2?????????@?
G__inference_doc_embedding_layer_call_and_return_conditional_losses_5700_/?,
%?"
 ?
inputs?????????

? ")?&
?
0?????????
@
? ?
,__inference_doc_embedding_layer_call_fn_5707R/?,
%?"
 ?
inputs?????????

? "??????????
@?
A__inference_doc_fc1_layer_call_and_return_conditional_losses_5738d3?0
)?&
$?!
inputs?????????
@
? ")?&
?
0?????????
@
? ?
&__inference_doc_fc1_layer_call_fn_5747W3?0
)?&
$?!
inputs?????????
@
? "??????????
@?
A__inference_doc_fc2_layer_call_and_return_conditional_losses_5799\)*/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? y
&__inference_doc_fc2_layer_call_fn_5808O)*/?,
%?"
 ?
inputs?????????@
? "??????????@?
?__inference_model_layer_call_and_return_conditional_losses_5451h)*8?5
.?+
!?
doc_ids?????????

p

 
? "%?"
?
0?????????@
? ?
?__inference_model_layer_call_and_return_conditional_losses_5472h)*8?5
.?+
!?
doc_ids?????????

p 

 
? "%?"
?
0?????????@
? ?
?__inference_model_layer_call_and_return_conditional_losses_5611g)*7?4
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
?__inference_model_layer_call_and_return_conditional_losses_5660g)*7?4
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
$__inference_model_layer_call_fn_5509[)*8?5
.?+
!?
doc_ids?????????

p

 
? "??????????@?
$__inference_model_layer_call_fn_5545[)*8?5
.?+
!?
doc_ids?????????

p 

 
? "??????????@?
$__inference_model_layer_call_fn_5675Z)*7?4
-?*
 ?
inputs?????????

p

 
? "??????????@?
$__inference_model_layer_call_fn_5690Z)*7?4
-?*
 ?
inputs?????????

p 

 
? "??????????@?
"__inference_signature_wrapper_5562w)*;?8
? 
1?.
,
doc_ids!?
doc_ids?????????
"1?.
,
doc_fc2!?
doc_fc2?????????@?
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5753d3?0
)?&
$?!
inputs?????????
@
? "-?*
#? 
0?????????
@
? ?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_5758W3?0
)?&
$?!
inputs?????????
@
? " ??????????
@?
M__inference_tf_op_layer_MaxPool_layer_call_and_return_conditional_losses_5763h7?4
-?*
(?%
inputs?????????
@
? "-?*
#? 
0?????????@
? ?
2__inference_tf_op_layer_MaxPool_layer_call_fn_5768[7?4
-?*
(?%
inputs?????????
@
? " ??????????@?
O__inference_tf_op_layer_Squeeze_1_layer_call_and_return_conditional_losses_5783\3?0
)?&
$?!
inputs?????????@
? "%?"
?
0?????????@
? ?
4__inference_tf_op_layer_Squeeze_1_layer_call_fn_5788O3?0
)?&
$?!
inputs?????????@
? "??????????@?
M__inference_tf_op_layer_Squeeze_layer_call_and_return_conditional_losses_5773d7?4
-?*
(?%
inputs?????????@
? ")?&
?
0?????????@
? ?
2__inference_tf_op_layer_Squeeze_layer_call_fn_5778W7?4
-?*
(?%
inputs?????????@
? "??????????@