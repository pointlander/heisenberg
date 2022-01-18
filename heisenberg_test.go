// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"
)

func BenchmarkMatrix64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := Matrix64{}.ControlledNot(8, []int{0}, 1)
		b := Matrix64{}.ControlledNot(8, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}

func BenchmarkMatrix128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := Matrix128{}.ControlledNot(8, []int{0}, 1)
		b := Matrix128{}.ControlledNot(8, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}

func BenchmarkSparse64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := Sparse64{}.ControlledNot(8, []int{0}, 1)
		b := Sparse64{}.ControlledNot(8, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}

func BenchmarkSparse128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		a := Sparse128{}.ControlledNot(8, []int{0}, 1)
		b := Sparse128{}.ControlledNot(8, []int{0, 1}, 2)
		c := a.Multiply(b)
		_ = c
	}
}
