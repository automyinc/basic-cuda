#!/bin/bash

VNX_INTERFACE_DIR=${VNX_INTERFACE_DIR:-/usr/interface}

BASIC_INCLUDE=$1
MATH_INCLUDE=$2
VNX_INCLUDE=$3

if [ -z "$BASIC_INCLUDE" ]; then
	BASIC_INCLUDE=${VNX_INTERFACE_DIR}/automy/basic/
fi

if [ -z "$MATH_INCLUDE" ]; then
	MATH_INCLUDE=${VNX_INTERFACE_DIR}/automy/math/
fi

if [ -z "$VNX_INCLUDE" ]; then
	VNX_INCLUDE=${VNX_INTERFACE_DIR}/vnx/
fi

cd $(dirname "$0")

vnxcppcodegen --cleanup generated/ automy.basic_cuda interface/ modules/ $BASIC_INCLUDE $MATH_INCLUDE $VNX_INCLUDE

