уп
в$Е$
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	АР
┤
ApplyRMSProp
var"TА

ms"TА
mom"TА
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
Д
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sigmoid
x"T
y"T"
Ttype:	
2
;
SigmoidGrad
x"T
y"T
z"T"
Ttype:	
2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.2.12
b'unknown'Щ■
l
xPlaceholder*+
_output_shapes
:         	Z*
dtype0* 
shape:         	Z
d
yPlaceholder*'
_output_shapes
:         Z*
dtype0*
shape:         Z
d
random_normal/shapeConst*
valueB"А   Z   *
dtype0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Я
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed2 *

seed *
T0*
_output_shapes
:	АZ*
dtype0
|
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes
:	АZ*
T0
e
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes
:	АZ*
T0
}
weights
VariableV2*
	container *
shared_name *
dtype0*
shape:	АZ*
_output_shapes
:	АZ
Я
weights/AssignAssignweightsrandom_normal*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
g
weights/readIdentityweights*
_output_shapes
:	АZ*
T0*
_class
loc:@weights
_
random_normal_1/shapeConst*
valueB:Z*
dtype0*
_output_shapes
:
Y
random_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_1/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ю
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2 *

seed *
T0*
_output_shapes
:Z*
dtype0
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
_output_shapes
:Z*
T0
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes
:Z*
T0
r
biases
VariableV2*
	container *
shared_name *
dtype0*
shape:Z*
_output_shapes
:Z
Щ
biases/AssignAssignbiasesrandom_normal_1*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
_
biases/readIdentitybiases*
_output_shapes
:Z*
T0*
_class
loc:@biases
°
unstackUnpackx*

axis*	
num	*
T0*┴
_output_shapesо
л:         Z:         Z:         Z:         Z:         Z:         Z:         Z:         Z:         Z
P
	rnn/ShapeShapeunstack*
_output_shapes
:*
T0*
out_type0
a
rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Н
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
ellipsis_mask *
shrink_axis_mask*

begin_mask *
_output_shapes
: *
new_axis_mask *
T0*
end_mask 
f
$rnn/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
 rnn/LSTMCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice$rnn/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
_output_shapes
:*
T0
f
rnn/LSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
c
!rnn/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
─
rnn/LSTMCellZeroState/concatConcatV2 rnn/LSTMCellZeroState/ExpandDimsrnn/LSTMCellZeroState/Const!rnn/LSTMCellZeroState/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
h
&rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
"rnn/LSTMCellZeroState/ExpandDims_1
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
_output_shapes
:*
T0
h
rnn/LSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
f
!rnn/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ч
rnn/LSTMCellZeroState/zerosFillrnn/LSTMCellZeroState/concat!rnn/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         А*
T0
h
&rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
"rnn/LSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
_output_shapes
:*
T0
h
rnn/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
e
#rnn/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╠
rnn/LSTMCellZeroState/concat_1ConcatV2"rnn/LSTMCellZeroState/ExpandDims_2rnn/LSTMCellZeroState/Const_2#rnn/LSTMCellZeroState/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
h
&rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ь
"rnn/LSTMCellZeroState/ExpandDims_3
ExpandDimsrnn/strided_slice&rnn/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
_output_shapes
:*
T0
h
rnn/LSTMCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
h
#rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
rnn/LSTMCellZeroState/zeros_1Fillrnn/LSTMCellZeroState/concat_1#rnn/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         А*
T0
п
5rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"┌      *
dtype0*'
_class
loc:@rnn/lstm_cell/kernel*
_output_shapes
:
б
3rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *╔л╣╜*
dtype0*'
_class
loc:@rnn/lstm_cell/kernel*
_output_shapes
: 
б
3rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *╔л╣=*
dtype0*'
_class
loc:@rnn/lstm_cell/kernel*
_output_shapes
: 
 
=rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform5rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed2 *

seed *'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А
ю
3rnn/lstm_cell/kernel/Initializer/random_uniform/subSub3rnn/lstm_cell/kernel/Initializer/random_uniform/max3rnn/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@rnn/lstm_cell/kernel
В
3rnn/lstm_cell/kernel/Initializer/random_uniform/mulMul=rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform3rnn/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
┌А*
T0*'
_class
loc:@rnn/lstm_cell/kernel
Ї
/rnn/lstm_cell/kernel/Initializer/random_uniformAdd3rnn/lstm_cell/kernel/Initializer/random_uniform/mul3rnn/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
┌А*
T0*'
_class
loc:@rnn/lstm_cell/kernel
╡
rnn/lstm_cell/kernel
VariableV2*
	container *
shared_name *
dtype0* 
_output_shapes
:
┌А*'
_class
loc:@rnn/lstm_cell/kernel*
shape:
┌А
щ
rnn/lstm_cell/kernel/AssignAssignrnn/lstm_cell/kernel/rnn/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
f
rnn/lstm_cell/kernel/readIdentityrnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
T0
s
1rnn/rnn/lstm_cell/lstm_cell/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
█
,rnn/rnn/lstm_cell/lstm_cell/lstm_cell/concatConcatV2unstackrnn/LSTMCellZeroState/zeros_11rnn/rnn/lstm_cell/lstm_cell/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
╪
,rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMulMatMul,rnn/rnn/lstm_cell/lstm_cell/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
Ъ
$rnn/lstm_cell/bias/Initializer/ConstConst*
valueBА*    *
dtype0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А
з
rnn/lstm_cell/bias
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes	
:А*%
_class
loc:@rnn/lstm_cell/bias*
shape:А
╙
rnn/lstm_cell/bias/AssignAssignrnn/lstm_cell/bias$rnn/lstm_cell/bias/Initializer/Const*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
]
rnn/lstm_cell/bias/readIdentityrnn/lstm_cell/bias*
_output_shapes	
:А*
T0
╔
-rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAddBiasAdd,rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
c
!rnn/rnn/lstm_cell/lstm_cell/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
m
+rnn/rnn/lstm_cell/lstm_cell/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
!rnn/rnn/lstm_cell/lstm_cell/splitSplit+rnn/rnn/lstm_cell/lstm_cell/split/split_dim-rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
f
!rnn/rnn/lstm_cell/lstm_cell/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
б
rnn/rnn/lstm_cell/lstm_cell/addAdd#rnn/rnn/lstm_cell/lstm_cell/split:2!rnn/rnn/lstm_cell/lstm_cell/add/y*(
_output_shapes
:         А*
T0
В
#rnn/rnn/lstm_cell/lstm_cell/SigmoidSigmoidrnn/rnn/lstm_cell/lstm_cell/add*(
_output_shapes
:         А*
T0
Ы
rnn/rnn/lstm_cell/lstm_cell/mulMul#rnn/rnn/lstm_cell/lstm_cell/Sigmoidrnn/LSTMCellZeroState/zeros*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1Sigmoid!rnn/rnn/lstm_cell/lstm_cell/split*(
_output_shapes
:         А*
T0
А
 rnn/rnn/lstm_cell/lstm_cell/TanhTanh#rnn/rnn/lstm_cell/lstm_cell/split:1*(
_output_shapes
:         А*
T0
д
!rnn/rnn/lstm_cell/lstm_cell/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1 rnn/rnn/lstm_cell/lstm_cell/Tanh*(
_output_shapes
:         А*
T0
Я
!rnn/rnn/lstm_cell/lstm_cell/add_1Addrnn/rnn/lstm_cell/lstm_cell/mul!rnn/rnn/lstm_cell/lstm_cell/mul_1*(
_output_shapes
:         А*
T0
И
%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2Sigmoid#rnn/rnn/lstm_cell/lstm_cell/split:3*(
_output_shapes
:         А*
T0
А
"rnn/rnn/lstm_cell/lstm_cell/Tanh_1Tanh!rnn/rnn/lstm_cell/lstm_cell/add_1*(
_output_shapes
:         А*
T0
ж
!rnn/rnn/lstm_cell/lstm_cell/mul_2Mul%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2"rnn/rnn/lstm_cell/lstm_cell/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
х
.rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concatConcatV2	unstack:1!rnn/rnn/lstm_cell/lstm_cell/mul_23rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_1/splitSplit-rnn/rnn/lstm_cell/lstm_cell_1/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_1/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_1/addAdd%rnn/rnn/lstm_cell/lstm_cell_1/split:2#rnn/rnn/lstm_cell/lstm_cell_1/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_1/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_1/add*(
_output_shapes
:         А*
T0
е
!rnn/rnn/lstm_cell/lstm_cell_1/mulMul%rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid!rnn/rnn/lstm_cell/lstm_cell/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_1/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_1/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_1/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_1/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_1/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_1/add_1Add!rnn/rnn/lstm_cell/lstm_cell_1/mul#rnn/rnn/lstm_cell/lstm_cell_1/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_1/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_1/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_1/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concatConcatV2	unstack:2#rnn/rnn/lstm_cell/lstm_cell_1/mul_23rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_2/splitSplit-rnn/rnn/lstm_cell/lstm_cell_2/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_2/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_2/addAdd%rnn/rnn/lstm_cell/lstm_cell_2/split:2#rnn/rnn/lstm_cell/lstm_cell_2/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_2/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_2/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_2/mulMul%rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_1/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_2/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_2/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_2/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_2/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_2/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_2/add_1Add!rnn/rnn/lstm_cell/lstm_cell_2/mul#rnn/rnn/lstm_cell/lstm_cell_2/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_2/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_2/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_2/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concatConcatV2	unstack:3#rnn/rnn/lstm_cell/lstm_cell_2/mul_23rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_3/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_3/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_3/splitSplit-rnn/rnn/lstm_cell/lstm_cell_3/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_3/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_3/addAdd%rnn/rnn/lstm_cell/lstm_cell_3/split:2#rnn/rnn/lstm_cell/lstm_cell_3/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_3/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_3/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_3/mulMul%rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_2/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_3/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_3/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_3/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_3/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_3/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_3/add_1Add!rnn/rnn/lstm_cell/lstm_cell_3/mul#rnn/rnn/lstm_cell/lstm_cell_3/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_3/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_3/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_3/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concatConcatV2	unstack:4#rnn/rnn/lstm_cell/lstm_cell_3/mul_23rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_4/splitSplit-rnn/rnn/lstm_cell/lstm_cell_4/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_4/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_4/addAdd%rnn/rnn/lstm_cell/lstm_cell_4/split:2#rnn/rnn/lstm_cell/lstm_cell_4/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_4/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_4/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_4/mulMul%rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_3/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_4/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_4/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_4/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_4/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_4/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_4/add_1Add!rnn/rnn/lstm_cell/lstm_cell_4/mul#rnn/rnn/lstm_cell/lstm_cell_4/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_4/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_4/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_4/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concatConcatV2	unstack:5#rnn/rnn/lstm_cell/lstm_cell_4/mul_23rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_5/splitSplit-rnn/rnn/lstm_cell/lstm_cell_5/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_5/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_5/addAdd%rnn/rnn/lstm_cell/lstm_cell_5/split:2#rnn/rnn/lstm_cell/lstm_cell_5/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_5/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_5/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_5/mulMul%rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_4/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_5/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_5/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_5/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_5/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_5/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_5/add_1Add!rnn/rnn/lstm_cell/lstm_cell_5/mul#rnn/rnn/lstm_cell/lstm_cell_5/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_5/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_5/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_5/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concatConcatV2	unstack:6#rnn/rnn/lstm_cell/lstm_cell_5/mul_23rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_6/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_6/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_6/splitSplit-rnn/rnn/lstm_cell/lstm_cell_6/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_6/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_6/addAdd%rnn/rnn/lstm_cell/lstm_cell_6/split:2#rnn/rnn/lstm_cell/lstm_cell_6/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_6/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_6/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_6/mulMul%rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_5/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_6/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_6/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_6/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_6/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_6/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_6/add_1Add!rnn/rnn/lstm_cell/lstm_cell_6/mul#rnn/rnn/lstm_cell/lstm_cell_6/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_6/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_6/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_6/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concatConcatV2	unstack:7#rnn/rnn/lstm_cell/lstm_cell_6/mul_23rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_7/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_7/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_7/splitSplit-rnn/rnn/lstm_cell/lstm_cell_7/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_7/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_7/addAdd%rnn/rnn/lstm_cell/lstm_cell_7/split:2#rnn/rnn/lstm_cell/lstm_cell_7/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_7/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_7/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_7/mulMul%rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_6/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_7/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_7/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_7/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_7/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_7/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_7/add_1Add!rnn/rnn/lstm_cell/lstm_cell_7/mul#rnn/rnn/lstm_cell/lstm_cell_7/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_7/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_7/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_7/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1*(
_output_shapes
:         А*
T0
u
3rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ч
.rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concatConcatV2	unstack:8#rnn/rnn/lstm_cell/lstm_cell_7/mul_23rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat/axis*(
_output_shapes
:         ┌*
T0*

Tidx0*
N
▄
.rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMulMatMul.rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concatrnn/lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
═
/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAddBiasAdd.rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMulrnn/lstm_cell/bias/read*(
_output_shapes
:         А*
data_formatNHWC*
T0
e
#rnn/rnn/lstm_cell/lstm_cell_8/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
o
-rnn/rnn/lstm_cell/lstm_cell_8/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
М
#rnn/rnn/lstm_cell/lstm_cell_8/splitSplit-rnn/rnn/lstm_cell/lstm_cell_8/split/split_dim/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:         А:         А:         А:         А*
T0
h
#rnn/rnn/lstm_cell/lstm_cell_8/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
з
!rnn/rnn/lstm_cell/lstm_cell_8/addAdd%rnn/rnn/lstm_cell/lstm_cell_8/split:2#rnn/rnn/lstm_cell/lstm_cell_8/add/y*(
_output_shapes
:         А*
T0
Ж
%rnn/rnn/lstm_cell/lstm_cell_8/SigmoidSigmoid!rnn/rnn/lstm_cell/lstm_cell_8/add*(
_output_shapes
:         А*
T0
з
!rnn/rnn/lstm_cell/lstm_cell_8/mulMul%rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid#rnn/rnn/lstm_cell/lstm_cell_7/add_1*(
_output_shapes
:         А*
T0
К
'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1Sigmoid#rnn/rnn/lstm_cell/lstm_cell_8/split*(
_output_shapes
:         А*
T0
Д
"rnn/rnn/lstm_cell/lstm_cell_8/TanhTanh%rnn/rnn/lstm_cell/lstm_cell_8/split:1*(
_output_shapes
:         А*
T0
к
#rnn/rnn/lstm_cell/lstm_cell_8/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1"rnn/rnn/lstm_cell/lstm_cell_8/Tanh*(
_output_shapes
:         А*
T0
е
#rnn/rnn/lstm_cell/lstm_cell_8/add_1Add!rnn/rnn/lstm_cell/lstm_cell_8/mul#rnn/rnn/lstm_cell/lstm_cell_8/mul_1*(
_output_shapes
:         А*
T0
М
'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2Sigmoid%rnn/rnn/lstm_cell/lstm_cell_8/split:3*(
_output_shapes
:         А*
T0
Д
$rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1Tanh#rnn/rnn/lstm_cell/lstm_cell_8/add_1*(
_output_shapes
:         А*
T0
м
#rnn/rnn/lstm_cell/lstm_cell_8/mul_2Mul'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2$rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1*(
_output_shapes
:         А*
T0
O
ShapeConst*
valueB:Z*
dtype0*
_output_shapes
:
Ы
muloutMatMul#rnn/rnn/lstm_cell/lstm_cell_8/mul_2weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         Z
R
predAddmuloutbiases/read*'
_output_shapes
:         Z*
T0
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
K
Shape_1Shapepred*
_output_shapes
:*
T0*
out_type0
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
K
Shape_2Shapepred*
_output_shapes
:*
T0*
out_type0
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*

axis *
_output_shapes
:*
T0*
N
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_2Slice/begin
Slice/size*
Index0*
_output_shapes
:*
T0
b
concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*
T0*

Tidx0*
N
i
ReshapeReshapepredconcat*0
_output_shapes
:                  *
Tshape0*
T0
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
H
Shape_3Shapey*
_output_shapes
:*
T0*
out_type0
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*

axis *
_output_shapes
:*
T0*
N
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_3Slice_1/beginSlice_1/size*
Index0*
_output_shapes
:*
T0
d
concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
j
	Reshape_1Reshapeyconcat_1*0
_output_shapes
:                  *
Tshape0*
T0
Ь
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*?
_output_shapes-
+:         :                  *
T0
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*

axis *
_output_shapes
:*
T0*
N
q
Slice_2SliceShape_1Slice_2/beginSlice_2/size*
Index0*#
_output_shapes
:         *
T0
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:         *
Tshape0*
T0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
\
MeanMean	Reshape_2Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_2*
_output_shapes
:*
T0*
out_type0
Ш
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:         *
T0*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
А
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
И
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:         *
T0
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
д
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*#
_output_shapes
:         *
Tshape0*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:                  *
T0
Ж
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:         *
T0
╠
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:                  *
T0
`
gradients/Reshape_grad/ShapeShapepred*
_output_shapes
:*
T0*
out_type0
╣
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*'
_output_shapes
:         Z*
Tshape0*
T0
_
gradients/pred_grad/ShapeShapemulout*
_output_shapes
:*
T0*
out_type0
e
gradients/pred_grad/Shape_1Const*
valueB:Z*
dtype0*
_output_shapes
:
╖
)gradients/pred_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pred_grad/Shapegradients/pred_grad/Shape_1*2
_output_shapes 
:         :         *
T0
й
gradients/pred_grad/SumSumgradients/Reshape_grad/Reshape)gradients/pred_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Ъ
gradients/pred_grad/ReshapeReshapegradients/pred_grad/Sumgradients/pred_grad/Shape*'
_output_shapes
:         Z*
Tshape0*
T0
н
gradients/pred_grad/Sum_1Sumgradients/Reshape_grad/Reshape+gradients/pred_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
У
gradients/pred_grad/Reshape_1Reshapegradients/pred_grad/Sum_1gradients/pred_grad/Shape_1*
_output_shapes
:Z*
Tshape0*
T0
j
$gradients/pred_grad/tuple/group_depsNoOp^gradients/pred_grad/Reshape^gradients/pred_grad/Reshape_1
▐
,gradients/pred_grad/tuple/control_dependencyIdentitygradients/pred_grad/Reshape%^gradients/pred_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*.
_class$
" loc:@gradients/pred_grad/Reshape
╫
.gradients/pred_grad/tuple/control_dependency_1Identitygradients/pred_grad/Reshape_1%^gradients/pred_grad/tuple/group_deps*
_output_shapes
:Z*
T0*0
_class&
$"loc:@gradients/pred_grad/Reshape_1
╗
gradients/mulout_grad/MatMulMatMul,gradients/pred_grad/tuple/control_dependencyweights/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         А
╦
gradients/mulout_grad/MatMul_1MatMul#rnn/rnn/lstm_cell/lstm_cell_8/mul_2,gradients/pred_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	АZ
n
&gradients/mulout_grad/tuple/group_depsNoOp^gradients/mulout_grad/MatMul^gradients/mulout_grad/MatMul_1
х
.gradients/mulout_grad/tuple/control_dependencyIdentitygradients/mulout_grad/MatMul'^gradients/mulout_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*/
_class%
#!loc:@gradients/mulout_grad/MatMul
т
0gradients/mulout_grad/tuple/control_dependency_1Identitygradients/mulout_grad/MatMul_1'^gradients/mulout_grad/tuple/group_deps*
_output_shapes
:	АZ*
T0*1
_class'
%#loc:@gradients/mulout_grad/MatMul_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╞
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/mulMul.gradients/mulout_grad/tuple/control_dependency$rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
╦
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2.gradients/mulout_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_8/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_8/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/SumSum<gradients/rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1_grad/TanhGradHgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Sum_1Sum<gradients/rnn/rnn/lstm_cell/lstm_cell_8/Tanh_1_grad/TanhGradJgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_7/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_7/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_8/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_8/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_8/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_8/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_8/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_8/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_8/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_8/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_8/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_8/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_8/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_8/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_8/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_8/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeShape	unstack:8*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeNShapeN	unstack:8#rnn/rnn/lstm_cell/lstm_cell_7/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
м
gradients/AddNAddNKgradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_7/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_7/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_7/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╫
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/SumSumgradients/AddNHgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
█
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Sum_1Sumgradients/AddNJgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_6/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_6/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_7/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_7/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_7/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_7/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_7/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_7/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_7/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_7/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_7/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_7/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_7/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_7/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_7/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_7/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeShape	unstack:7*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeNShapeN	unstack:7#rnn/rnn/lstm_cell/lstm_cell_6/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_1AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_6/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_7/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_6/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_6/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/SumSumgradients/AddN_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Sum_1Sumgradients/AddN_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_5/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_5/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_6/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_6/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_6/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_6/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_6/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_6/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_6/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_6/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_6/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_6/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_6/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_6/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_6/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_6/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeShape	unstack:6*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeNShapeN	unstack:6#rnn/rnn/lstm_cell/lstm_cell_5/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_2AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_5/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_6/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_5/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_5/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/SumSumgradients/AddN_2Hgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Sum_1Sumgradients/AddN_2Jgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_4/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_4/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_5/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_5/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_5/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_5/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_5/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_5/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_5/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_5/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_5/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_5/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_5/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_5/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_5/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_5/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeShape	unstack:5*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeNShapeN	unstack:5#rnn/rnn/lstm_cell/lstm_cell_4/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_3AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_4/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_5/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_4/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_4/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/SumSumgradients/AddN_3Hgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Sum_1Sumgradients/AddN_3Jgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_3/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_3/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_4/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_4/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_4/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_4/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_4/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_4/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_4/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_4/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_4/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_4/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_4/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_4/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_4/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_4/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeShape	unstack:4*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeNShapeN	unstack:4#rnn/rnn/lstm_cell/lstm_cell_3/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_4AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_3/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_4/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_3/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_3/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/SumSumgradients/AddN_4Hgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Sum_1Sumgradients/AddN_4Jgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_2/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_2/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_3/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_3/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_3/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_3/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_3/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_3/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_3/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_3/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_3/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_3/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_3/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_3/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_3/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_3/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeShape	unstack:3*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeNShapeN	unstack:3#rnn/rnn/lstm_cell/lstm_cell_2/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_5AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_2/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_3/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_2/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_2/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/SumSumgradients/AddN_5Hgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Sum_1Sumgradients/AddN_5Jgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid*
_output_shapes
:*
T0*
out_type0
Ы
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_1/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
р
4gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependency#rnn/rnn/lstm_cell/lstm_cell_1/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_2/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_2/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_2/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_2/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_2/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_2/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_2/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_2/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_2/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_2/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_2/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_2/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_2/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_2/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeShape	unstack:2*
_output_shapes
:*
T0*
out_type0
┬
Dgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeNShapeN	unstack:2#rnn/rnn/lstm_cell/lstm_cell_1/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/Slice_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ю
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Shape_1Shape$rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ё
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/control_dependency_1$rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ї
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/Reshape_1
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ё
<gradients/rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1_grad/TanhGradTanhGrad$rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
о
gradients/AddN_6AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/tuple/control_dependency_1<gradients/rnn/rnn/lstm_cell/lstm_cell_1/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_2/mul_grad/Reshape_1*
N
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/ShapeShape!rnn/rnn/lstm_cell/lstm_cell_1/mul*
_output_shapes
:*
T0*
out_type0
Э
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Shape_1Shape#rnn/rnn/lstm_cell/lstm_cell_1/mul_1*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/SumSumgradients/AddN_6Hgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▌
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Sum_1Sumgradients/AddN_6Jgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Shape_1Shape!rnn/rnn/lstm_cell/lstm_cell/add_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▐
4gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependency!rnn/rnn/lstm_cell/lstm_cell/add_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell_1/SigmoidKgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape_1
Я
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/ShapeShape'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell_1/Tanh*
_output_shapes
:*
T0*
out_type0
Ф
Hgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Shape:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
у
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/mulMulMgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell_1/Tanh*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/SumSum6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/mulHgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/ReshapeReshape6gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/mul_1Mul'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1Mgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Е
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Sum_1Sum8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/mul_1Jgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
■
<gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape_1Reshape8gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Sum_1:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╟
Cgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/group_depsNoOp;^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape=^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape_1
█
Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/control_dependencyIdentity:gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/ReshapeD^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape
с
Mgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/control_dependency_1Identity<gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape_1D^gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*O
_classE
CAloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell_1/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
·
Bgradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1_grad/SigmoidGradSigmoidGrad'rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ь
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/Tanh_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell_1/TanhMgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell_1/split:2*
_output_shapes
:*
T0*
out_type0
{
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Е
4gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/SumSum@gradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Й
6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Sum_1Sum@gradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_grad/SigmoidGradHgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ц
:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape
╟
Kgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/group_deps*
_output_shapes
: *
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/Reshape_1
╦
9gradients/rnn/rnn/lstm_cell/lstm_cell_1/split_grad/concatConcatV2Bgradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_1_grad/SigmoidGrad:gradients/rnn/rnn/lstm_cell/lstm_cell_1/Tanh_grad/TanhGradIgradients/rnn/rnn/lstm_cell/lstm_cell_1/add_grad/tuple/control_dependencyBgradients/rnn/rnn/lstm_cell/lstm_cell_1/Sigmoid_2_grad/SigmoidGrad-rnn/rnn/lstm_cell/lstm_cell_1/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
╤
Jgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/rnn/rnn/lstm_cell/lstm_cell_1/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
р
Ogradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp:^gradients/rnn/rnn/lstm_cell/lstm_cell_1/split_grad/concatK^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad
ё
Wgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9gradients/rnn/rnn/lstm_cell/lstm_cell_1/split_grad/concatP^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*L
_classB
@>loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/split_grad/concat
И
Ygradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/BiasAddGradP^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad
Ы
Dgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMulMatMulWgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
к
Fgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMul_1MatMul.rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concatWgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
ц
Ngradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMulG^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMul_1
Е
Vgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMulO^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMul
Г
Xgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMul_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
ч
Agradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/modFloorMod3rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat/axisBgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
М
Cgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeShape	unstack:1*
_output_shapes
:*
T0*
out_type0
└
Dgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeNShapeN	unstack:1!rnn/rnn/lstm_cell/lstm_cell/mul_2* 
_output_shapes
::*
T0*
N*
out_type0
╓
Jgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ConcatOffsetConcatOffsetAgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/modDgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeNFgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
■
Cgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/SliceSliceVgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyJgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ConcatOffsetDgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeN*
Index0*0
_output_shapes
:                  *
T0
Д
Egradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Slice_1SliceVgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyLgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ConcatOffset:1Fgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/ShapeN:1*
Index0*0
_output_shapes
:                  *
T0
ф
Ngradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/group_depsNoOpD^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/SliceF^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Slice_1
В
Vgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/control_dependencyIdentityCgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/SliceO^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/group_deps*'
_output_shapes
:         Z*
T0*V
_classL
JHloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Slice
Й
Xgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/control_dependency_1IdentityEgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Slice_1O^gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*X
_classN
LJloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/Slice_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
Ъ
8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Shape_1Shape"rnn/rnn/lstm_cell/lstm_cell/Tanh_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Shape_1*2
_output_shapes 
:         :         *
T0
ь
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/mulMulXgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/control_dependency_1"rnn/rnn/lstm_cell/lstm_cell/Tanh_1*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ё
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2Xgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/concat_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/Reshape_1
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2Igradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ъ
:gradients/rnn/rnn/lstm_cell/lstm_cell/Tanh_1_grad/TanhGradTanhGrad"rnn/rnn/lstm_cell/lstm_cell/Tanh_1Kgradients/rnn/rnn/lstm_cell/lstm_cell/mul_2_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
м
gradients/AddN_7AddNKgradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/tuple/control_dependency_1:gradients/rnn/rnn/lstm_cell/lstm_cell/Tanh_1_grad/TanhGrad*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell_1/mul_grad/Reshape_1*
N
Х
6gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/ShapeShapernn/rnn/lstm_cell/lstm_cell/mul*
_output_shapes
:*
T0*
out_type0
Щ
8gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Shape_1Shape!rnn/rnn/lstm_cell/lstm_cell/mul_1*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╒
4gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/SumSumgradients/AddN_7Fgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
┘
6gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_7Hgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/Reshape_1
Ч
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/ShapeShape#rnn/rnn/lstm_cell/lstm_cell/Sigmoid*
_output_shapes
:*
T0*
out_type0
С
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Shape_1Shapernn/LSTMCellZeroState/zeros*
_output_shapes
:*
T0*
out_type0
И
Dgradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Shape6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╘
2gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/mulMulIgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependencyrnn/LSTMCellZeroState/zeros*(
_output_shapes
:         А*
T0
є
2gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/SumSum2gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/mulDgradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ь
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/ReshapeReshape2gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Sum4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
▐
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/mul_1Mul#rnn/rnn/lstm_cell/lstm_cell/SigmoidIgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Sum_1Sum4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/mul_1Fgradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape_1Reshape4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Sum_16gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
╗
?gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/group_depsNoOp7^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape9^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape_1
╦
Ggradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/control_dependencyIdentity6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape@^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*I
_class?
=;loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape
╤
Igradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/control_dependency_1Identity8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape_1@^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/Reshape_1
Ы
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/ShapeShape%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1*
_output_shapes
:*
T0*
out_type0
Ш
8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Shape_1Shape rnn/rnn/lstm_cell/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
О
Fgradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Shape8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
▌
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/mulMulKgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependency_1 rnn/rnn/lstm_cell/lstm_cell/Tanh*(
_output_shapes
:         А*
T0
∙
4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/SumSum4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/mulFgradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
Є
8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/ReshapeReshape4gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Sum6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
ф
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/mul_1Mul%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1Kgradients/rnn/rnn/lstm_cell/lstm_cell/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
 
6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Sum_1Sum6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/mul_1Hgradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
°
:gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape_1Reshape6gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Sum_18gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Shape_1*(
_output_shapes
:         А*
Tshape0*
T0
┴
Agradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/group_depsNoOp9^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape;^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape_1
╙
Igradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/control_dependencyIdentity8gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/ReshapeB^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape
┘
Kgradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/control_dependency_1Identity:gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape_1B^gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*M
_classC
A?loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/Reshape_1
ю
>gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad#rnn/rnn/lstm_cell/lstm_cell/SigmoidGgradients/rnn/rnn/lstm_cell/lstm_cell/mul_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Ї
@gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad%rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1Igradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ц
8gradients/rnn/rnn/lstm_cell/lstm_cell/Tanh_grad/TanhGradTanhGrad rnn/rnn/lstm_cell/lstm_cell/TanhKgradients/rnn/rnn/lstm_cell/lstm_cell/mul_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
Ч
4gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/ShapeShape#rnn/rnn/lstm_cell/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0
y
6gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
И
Dgradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Shape6gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
 
2gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/SumSum>gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_grad/SigmoidGradDgradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
ь
6gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/ReshapeReshape2gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Sum4gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Shape*(
_output_shapes
:         А*
Tshape0*
T0
Г
4gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Sum_1Sum>gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_grad/SigmoidGradFgradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
р
8gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape_1Reshape4gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Sum_16gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
╗
?gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/group_depsNoOp7^gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape9^gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape_1
╦
Ggradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/control_dependencyIdentity6gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape@^gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*I
_class?
=;loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape
┐
Igradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/control_dependency_1Identity8gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape_1@^gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/group_deps*
_output_shapes
: *
T0*K
_classA
?=loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/Reshape_1
┐
7gradients/rnn/rnn/lstm_cell/lstm_cell/split_grad/concatConcatV2@gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_1_grad/SigmoidGrad8gradients/rnn/rnn/lstm_cell/lstm_cell/Tanh_grad/TanhGradGgradients/rnn/rnn/lstm_cell/lstm_cell/add_grad/tuple/control_dependency@gradients/rnn/rnn/lstm_cell/lstm_cell/Sigmoid_2_grad/SigmoidGrad+rnn/rnn/lstm_cell/lstm_cell/split/split_dim*(
_output_shapes
:         А*
T0*

Tidx0*
N
═
Hgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/rnn/rnn/lstm_cell/lstm_cell/split_grad/concat*
_output_shapes	
:А*
data_formatNHWC*
T0
┌
Mgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/group_depsNoOp8^gradients/rnn/rnn/lstm_cell/lstm_cell/split_grad/concatI^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/BiasAddGrad
щ
Ugradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/rnn/rnn/lstm_cell/lstm_cell/split_grad/concatN^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*J
_class@
><loc:@gradients/rnn/rnn/lstm_cell/lstm_cell/split_grad/concat
А
Wgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/BiasAddGradN^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*[
_classQ
OMloc:@gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/BiasAddGrad
Ч
Bgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMulMatMulUgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/lstm_cell/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         ┌
д
Dgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMul_1MatMul,rnn/rnn/lstm_cell/lstm_cell/lstm_cell/concatUgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
┌А
р
Lgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/group_depsNoOpC^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMulE^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMul_1
¤
Tgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityBgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMulM^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         ┌*
T0*U
_classK
IGloc:@gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMul
√
Vgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityDgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMul_1M^gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
┌А*
T0*W
_classM
KIloc:@gradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/MatMul_1
╫
gradients/AddN_8AddNYgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1Wgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/BiasAdd_grad/BiasAddGrad*
N	
╧
gradients/AddN_9AddNXgradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_7/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_6/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_5/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_4/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_3/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_2/lstm_cell/MatMul_grad/tuple/control_dependency_1Xgradients/rnn/rnn/lstm_cell/lstm_cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1Vgradients/rnn/rnn/lstm_cell/lstm_cell/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
┌А*
T0*Y
_classO
MKloc:@gradients/rnn/rnn/lstm_cell/lstm_cell_8/lstm_cell/MatMul_grad/MatMul_1*
N	
У
 weights/RMSProp/Initializer/onesConst*
valueB	АZ*  А?*
dtype0*
_class
loc:@weights*
_output_shapes
:	АZ
б
weights/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	АZ*
_class
loc:@weights*
shape:	АZ
┬
weights/RMSProp/AssignAssignweights/RMSProp weights/RMSProp/Initializer/ones*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
w
weights/RMSProp/readIdentityweights/RMSProp*
_output_shapes
:	АZ*
T0*
_class
loc:@weights
Ц
#weights/RMSProp_1/Initializer/zerosConst*
valueB	АZ*    *
dtype0*
_class
loc:@weights*
_output_shapes
:	АZ
г
weights/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:	АZ*
_class
loc:@weights*
shape:	АZ
╔
weights/RMSProp_1/AssignAssignweights/RMSProp_1#weights/RMSProp_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
{
weights/RMSProp_1/readIdentityweights/RMSProp_1*
_output_shapes
:	АZ*
T0*
_class
loc:@weights
З
biases/RMSProp/Initializer/onesConst*
valueBZ*  А?*
dtype0*
_class
loc:@biases*
_output_shapes
:Z
Х
biases/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:Z*
_class
loc:@biases*
shape:Z
╣
biases/RMSProp/AssignAssignbiases/RMSPropbiases/RMSProp/Initializer/ones*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
o
biases/RMSProp/readIdentitybiases/RMSProp*
_output_shapes
:Z*
T0*
_class
loc:@biases
К
"biases/RMSProp_1/Initializer/zerosConst*
valueBZ*    *
dtype0*
_class
loc:@biases*
_output_shapes
:Z
Ч
biases/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:Z*
_class
loc:@biases*
shape:Z
└
biases/RMSProp_1/AssignAssignbiases/RMSProp_1"biases/RMSProp_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
s
biases/RMSProp_1/readIdentitybiases/RMSProp_1*
_output_shapes
:Z*
T0*
_class
loc:@biases
п
-rnn/lstm_cell/kernel/RMSProp/Initializer/onesConst*
valueB
┌А*  А?*
dtype0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А
╜
rnn/lstm_cell/kernel/RMSProp
VariableV2*
	container *
shared_name *
dtype0* 
_output_shapes
:
┌А*'
_class
loc:@rnn/lstm_cell/kernel*
shape:
┌А
ў
#rnn/lstm_cell/kernel/RMSProp/AssignAssignrnn/lstm_cell/kernel/RMSProp-rnn/lstm_cell/kernel/RMSProp/Initializer/ones*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
Я
!rnn/lstm_cell/kernel/RMSProp/readIdentityrnn/lstm_cell/kernel/RMSProp* 
_output_shapes
:
┌А*
T0*'
_class
loc:@rnn/lstm_cell/kernel
▓
0rnn/lstm_cell/kernel/RMSProp_1/Initializer/zerosConst*
valueB
┌А*    *
dtype0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А
┐
rnn/lstm_cell/kernel/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0* 
_output_shapes
:
┌А*'
_class
loc:@rnn/lstm_cell/kernel*
shape:
┌А
■
%rnn/lstm_cell/kernel/RMSProp_1/AssignAssignrnn/lstm_cell/kernel/RMSProp_10rnn/lstm_cell/kernel/RMSProp_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
г
#rnn/lstm_cell/kernel/RMSProp_1/readIdentityrnn/lstm_cell/kernel/RMSProp_1* 
_output_shapes
:
┌А*
T0*'
_class
loc:@rnn/lstm_cell/kernel
б
+rnn/lstm_cell/bias/RMSProp/Initializer/onesConst*
valueBА*  А?*
dtype0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А
п
rnn/lstm_cell/bias/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes	
:А*%
_class
loc:@rnn/lstm_cell/bias*
shape:А
ъ
!rnn/lstm_cell/bias/RMSProp/AssignAssignrnn/lstm_cell/bias/RMSProp+rnn/lstm_cell/bias/RMSProp/Initializer/ones*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
Ф
rnn/lstm_cell/bias/RMSProp/readIdentityrnn/lstm_cell/bias/RMSProp*
_output_shapes	
:А*
T0*%
_class
loc:@rnn/lstm_cell/bias
д
.rnn/lstm_cell/bias/RMSProp_1/Initializer/zerosConst*
valueBА*    *
dtype0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А
▒
rnn/lstm_cell/bias/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes	
:А*%
_class
loc:@rnn/lstm_cell/bias*
shape:А
ё
#rnn/lstm_cell/bias/RMSProp_1/AssignAssignrnn/lstm_cell/bias/RMSProp_1.rnn/lstm_cell/bias/RMSProp_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
Ш
!rnn/lstm_cell/bias/RMSProp_1/readIdentityrnn/lstm_cell/bias/RMSProp_1*
_output_shapes	
:А*
T0*%
_class
loc:@rnn/lstm_cell/bias
Z
RMSProp/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
R
RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
RMSProp/epsilonConst*
valueB
 * ц█.*
dtype0*
_output_shapes
: 
┤
#RMSProp/update_weights/ApplyRMSPropApplyRMSPropweightsweights/RMSPropweights/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon0gradients/mulout_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@weights*
_output_shapes
:	АZ
и
"RMSProp/update_biases/ApplyRMSPropApplyRMSPropbiasesbiases/RMSPropbiases/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon.gradients/pred_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@biases*
_output_shapes
:Z
╓
0RMSProp/update_rnn/lstm_cell/kernel/ApplyRMSPropApplyRMSProprnn/lstm_cell/kernelrnn/lstm_cell/kernel/RMSProprnn/lstm_cell/kernel/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients/AddN_9*
use_locking( *
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А
╟
.RMSProp/update_rnn/lstm_cell/bias/ApplyRMSPropApplyRMSProprnn/lstm_cell/biasrnn/lstm_cell/bias/RMSProprnn/lstm_cell/bias/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilongradients/AddN_8*
use_locking( *
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А
╛
RMSPropNoOp$^RMSProp/update_weights/ApplyRMSProp#^RMSProp/update_biases/ApplyRMSProp1^RMSProp/update_rnn/lstm_cell/kernel/ApplyRMSProp/^RMSProp/update_rnn/lstm_cell/bias/ApplyRMSProp
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
b
ArgMaxArgMaxpredArgMax/dimension*#
_output_shapes
:         *
T0*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
c
ArgMax_1ArgMaxyArgMax_1/dimension*#
_output_shapes
:         *
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:         *
T0	
R
Cast_1CastEqual*#
_output_shapes
:         *

DstT0*

SrcT0

Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
]
Mean_1MeanCast_1Const_1*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
х
initNoOp^weights/Assign^biases/Assign^rnn/lstm_cell/kernel/Assign^rnn/lstm_cell/bias/Assign^weights/RMSProp/Assign^weights/RMSProp_1/Assign^biases/RMSProp/Assign^biases/RMSProp_1/Assign$^rnn/lstm_cell/kernel/RMSProp/Assign&^rnn/lstm_cell/kernel/RMSProp_1/Assign"^rnn/lstm_cell/bias/RMSProp/Assign$^rnn/lstm_cell/bias/RMSProp_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
█
save/SaveV2/tensor_namesConst*О
valueДBБBbiasesBbiases/RMSPropBbiases/RMSProp_1Brnn/lstm_cell/biasBrnn/lstm_cell/bias/RMSPropBrnn/lstm_cell/bias/RMSProp_1Brnn/lstm_cell/kernelBrnn/lstm_cell/kernel/RMSPropBrnn/lstm_cell/kernel/RMSProp_1BweightsBweights/RMSPropBweights/RMSProp_1*
dtype0*
_output_shapes
:
{
save/SaveV2/shape_and_slicesConst*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ю
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiasesbiases/RMSPropbiases/RMSProp_1rnn/lstm_cell/biasrnn/lstm_cell/bias/RMSProprnn/lstm_cell/bias/RMSProp_1rnn/lstm_cell/kernelrnn/lstm_cell/kernel/RMSProprnn/lstm_cell/kernel/RMSProp_1weightsweights/RMSPropweights/RMSProp_1*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
j
save/RestoreV2/tensor_namesConst*
valueBBbiases*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
Ц
save/AssignAssignbiasessave/RestoreV2*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
t
save/RestoreV2_1/tensor_namesConst*#
valueBBbiases/RMSProp*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
в
save/Assign_1Assignbiases/RMSPropsave/RestoreV2_1*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
v
save/RestoreV2_2/tensor_namesConst*%
valueBBbiases/RMSProp_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
д
save/Assign_2Assignbiases/RMSProp_1save/RestoreV2_2*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
x
save/RestoreV2_3/tensor_namesConst*'
valueBBrnn/lstm_cell/bias*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
│
save/Assign_3Assignrnn/lstm_cell/biassave/RestoreV2_3*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
А
save/RestoreV2_4/tensor_namesConst*/
value&B$Brnn/lstm_cell/bias/RMSProp*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
╗
save/Assign_4Assignrnn/lstm_cell/bias/RMSPropsave/RestoreV2_4*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
В
save/RestoreV2_5/tensor_namesConst*1
value(B&Brnn/lstm_cell/bias/RMSProp_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
╜
save/Assign_5Assignrnn/lstm_cell/bias/RMSProp_1save/RestoreV2_5*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
z
save/RestoreV2_6/tensor_namesConst*)
value BBrnn/lstm_cell/kernel*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
╝
save/Assign_6Assignrnn/lstm_cell/kernelsave/RestoreV2_6*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
В
save/RestoreV2_7/tensor_namesConst*1
value(B&Brnn/lstm_cell/kernel/RMSProp*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
─
save/Assign_7Assignrnn/lstm_cell/kernel/RMSPropsave/RestoreV2_7*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
Д
save/RestoreV2_8/tensor_namesConst*3
value*B(Brnn/lstm_cell/kernel/RMSProp_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
╞
save/Assign_8Assignrnn/lstm_cell/kernel/RMSProp_1save/RestoreV2_8*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
m
save/RestoreV2_9/tensor_namesConst*
valueBBweights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
б
save/Assign_9Assignweightssave/RestoreV2_9*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
v
save/RestoreV2_10/tensor_namesConst*$
valueBBweights/RMSProp*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
л
save/Assign_10Assignweights/RMSPropsave/RestoreV2_10*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
x
save/RestoreV2_11/tensor_namesConst*&
valueBBweights/RMSProp_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Щ
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
н
save/Assign_11Assignweights/RMSProp_1save/RestoreV2_11*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
╪
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_2ca6be7c381a40cbb4c5df14d5b4cc79/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
▌
save_1/SaveV2/tensor_namesConst*О
valueДBБBbiasesBbiases/RMSPropBbiases/RMSProp_1Brnn/lstm_cell/biasBrnn/lstm_cell/bias/RMSPropBrnn/lstm_cell/bias/RMSProp_1Brnn/lstm_cell/kernelBrnn/lstm_cell/kernel/RMSPropBrnn/lstm_cell/kernel/RMSProp_1BweightsBweights/RMSPropBweights/RMSProp_1*
dtype0*
_output_shapes
:
}
save_1/SaveV2/shape_and_slicesConst*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
А
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbiasesbiases/RMSPropbiases/RMSProp_1rnn/lstm_cell/biasrnn/lstm_cell/bias/RMSProprnn/lstm_cell/bias/RMSProp_1rnn/lstm_cell/kernelrnn/lstm_cell/kernel/RMSProprnn/lstm_cell/kernel/RMSProp_1weightsweights/RMSPropweights/RMSProp_1*
dtypes
2
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_1/ShardedFilename
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
_output_shapes
:*
T0*
N
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
_output_shapes
: *
T0
l
save_1/RestoreV2/tensor_namesConst*
valueBBbiases*
dtype0*
_output_shapes
:
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ш
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2
Ъ
save_1/AssignAssignbiasessave_1/RestoreV2*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
v
save_1/RestoreV2_1/tensor_namesConst*#
valueBBbiases/RMSProp*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2
ж
save_1/Assign_1Assignbiases/RMSPropsave_1/RestoreV2_1*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
x
save_1/RestoreV2_2/tensor_namesConst*%
valueBBbiases/RMSProp_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
и
save_1/Assign_2Assignbiases/RMSProp_1save_1/RestoreV2_2*
use_locking(*
T0*
_class
loc:@biases*
_output_shapes
:Z*
validate_shape(
z
save_1/RestoreV2_3/tensor_namesConst*'
valueBBrnn/lstm_cell/bias*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
_output_shapes
:*
dtypes
2
╖
save_1/Assign_3Assignrnn/lstm_cell/biassave_1/RestoreV2_3*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
В
save_1/RestoreV2_4/tensor_namesConst*/
value&B$Brnn/lstm_cell/bias/RMSProp*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
┐
save_1/Assign_4Assignrnn/lstm_cell/bias/RMSPropsave_1/RestoreV2_4*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
Д
save_1/RestoreV2_5/tensor_namesConst*1
value(B&Brnn/lstm_cell/bias/RMSProp_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
┴
save_1/Assign_5Assignrnn/lstm_cell/bias/RMSProp_1save_1/RestoreV2_5*
use_locking(*
T0*%
_class
loc:@rnn/lstm_cell/bias*
_output_shapes	
:А*
validate_shape(
|
save_1/RestoreV2_6/tensor_namesConst*)
value BBrnn/lstm_cell/kernel*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
└
save_1/Assign_6Assignrnn/lstm_cell/kernelsave_1/RestoreV2_6*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
Д
save_1/RestoreV2_7/tensor_namesConst*1
value(B&Brnn/lstm_cell/kernel/RMSProp*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
╚
save_1/Assign_7Assignrnn/lstm_cell/kernel/RMSPropsave_1/RestoreV2_7*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
Ж
save_1/RestoreV2_8/tensor_namesConst*3
value*B(Brnn/lstm_cell/kernel/RMSProp_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
╩
save_1/Assign_8Assignrnn/lstm_cell/kernel/RMSProp_1save_1/RestoreV2_8*
use_locking(*
T0*'
_class
loc:@rnn/lstm_cell/kernel* 
_output_shapes
:
┌А*
validate_shape(
o
save_1/RestoreV2_9/tensor_namesConst*
valueBBweights*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ю
save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
_output_shapes
:*
dtypes
2
е
save_1/Assign_9Assignweightssave_1/RestoreV2_9*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
x
 save_1/RestoreV2_10/tensor_namesConst*$
valueBBweights/RMSProp*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
б
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
п
save_1/Assign_10Assignweights/RMSPropsave_1/RestoreV2_10*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
z
 save_1/RestoreV2_11/tensor_namesConst*&
valueBBweights/RMSProp_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
б
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
▒
save_1/Assign_11Assignweights/RMSProp_1save_1/RestoreV2_11*
use_locking(*
T0*
_class
loc:@weights*
_output_shapes
:	АZ*
validate_shape(
Ї
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"
train_op
	
RMSProp"Ф
trainable_variables№∙
+
	weights:0weights/Assignweights/read:0
(
biases:0biases/Assignbiases/read:0
R
rnn/lstm_cell/kernel:0rnn/lstm_cell/kernel/Assignrnn/lstm_cell/kernel/read:0
L
rnn/lstm_cell/bias:0rnn/lstm_cell/bias/Assignrnn/lstm_cell/bias/read:0"╘
	variables╞├
+
	weights:0weights/Assignweights/read:0
(
biases:0biases/Assignbiases/read:0
R
rnn/lstm_cell/kernel:0rnn/lstm_cell/kernel/Assignrnn/lstm_cell/kernel/read:0
L
rnn/lstm_cell/bias:0rnn/lstm_cell/bias/Assignrnn/lstm_cell/bias/read:0
C
weights/RMSProp:0weights/RMSProp/Assignweights/RMSProp/read:0
I
weights/RMSProp_1:0weights/RMSProp_1/Assignweights/RMSProp_1/read:0
@
biases/RMSProp:0biases/RMSProp/Assignbiases/RMSProp/read:0
F
biases/RMSProp_1:0biases/RMSProp_1/Assignbiases/RMSProp_1/read:0
j
rnn/lstm_cell/kernel/RMSProp:0#rnn/lstm_cell/kernel/RMSProp/Assign#rnn/lstm_cell/kernel/RMSProp/read:0
p
 rnn/lstm_cell/kernel/RMSProp_1:0%rnn/lstm_cell/kernel/RMSProp_1/Assign%rnn/lstm_cell/kernel/RMSProp_1/read:0
d
rnn/lstm_cell/bias/RMSProp:0!rnn/lstm_cell/bias/RMSProp/Assign!rnn/lstm_cell/bias/RMSProp/read:0
j
rnn/lstm_cell/bias/RMSProp_1:0#rnn/lstm_cell/bias/RMSProp_1/Assign#rnn/lstm_cell/bias/RMSProp_1/read:0"$
legacy_init_op

legacy_init_op*В
predict_patternsn
*
patterns
x:0         	Z$
scores
y:0         Ztensorflow/serving/predict*и
serving_defaultФ
(
inputs
x:0         	Z$
scores
y:0         Z%
classes
y:0         Ztensorflow/serving/classify