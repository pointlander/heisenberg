// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"
)

func BenchmarkDense64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineDense64{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]int{0}, 1)
		machine.ControlledNot([]int{0, 1}, 2)
	}
}

func BenchmarkDense128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineDense128{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]int{0}, 1)
		machine.ControlledNot([]int{0, 1}, 2)
	}
}

func BenchmarkSparse64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineSparse64{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]int{0}, 1)
		machine.ControlledNot([]int{0, 1}, 2)
	}
}

func BenchmarkSparse128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineSparse128{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]int{0}, 1)
		machine.ControlledNot([]int{0, 1}, 2)
	}
}

func TestTensor64(t *testing.T) {
	p := &Sparse64{
		R: 2,
		C: 2,
		Matrix: []map[int]complex64{
			map[int]complex64{
				0: 1,
				1: 0,
			},
			map[int]complex64{
				0: 0,
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < 3; i++ {
		q = p.Tensor(q)
	}

	m := &Dense64{
		R: 2,
		C: 2,
		Matrix: []complex64{
			1, 0,
			0, 1,
		},
	}
	n := m
	for i := 0; i < 3; i++ {
		n = m.Tensor(n)
	}

	for i := 0; i < n.R; i += n.C {
		a := q.Matrix[i]
		for j := 0; j < n.C; j++ {
			var value complex64
			if a != nil {
				value = a[j]
			}
			if value != n.Matrix[i*n.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, n.Matrix[i*n.C+j])
			}
		}
	}
}

func TestTensor128(t *testing.T) {
	p := &Sparse128{
		R: 2,
		C: 2,
		Matrix: []map[int]complex128{
			map[int]complex128{
				0: 1,
				1: 0,
			},
			map[int]complex128{
				0: 0,
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < 3; i++ {
		q = p.Tensor(q)
	}

	m := &Dense128{
		R: 2,
		C: 2,
		Matrix: []complex128{
			1, 0,
			0, 1,
		},
	}
	n := m
	for i := 0; i < 3; i++ {
		n = m.Tensor(n)
	}

	for i := 0; i < n.R; i += n.C {
		a := q.Matrix[i]
		for j := 0; j < n.C; j++ {
			var value complex128
			if a != nil {
				value = a[j]
			}
			if value != n.Matrix[i*n.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, n.Matrix[i*n.C+j])
			}
		}
	}
}

func TestMultiply64(t *testing.T) {
	machineDense := MachineDense64{}
	machineDense.One()
	machineDense.Zero()
	machineDense.Zero()
	machineDense.ControlledNot([]int{0}, 1)
	machineDense.ControlledNot([]int{0, 1}, 2)

	machineSparse := MachineSparse64{}
	machineSparse.One()
	machineSparse.Zero()
	machineSparse.Zero()
	machineSparse.ControlledNot([]int{0}, 1)
	machineSparse.ControlledNot([]int{0, 1}, 2)

	for i := 0; i < machineDense.R; i += machineDense.C {
		a := machineSparse.Matrix[i]
		for j := 0; j < machineDense.C; j++ {
			var value complex64
			if a != nil {
				value = a[j]
			}
			if value != machineDense.Matrix[i*machineDense.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, machineDense.Matrix[i*machineDense.C+j])
			}
		}
	}
}

func TestMultiply128(t *testing.T) {
	machineDense := MachineDense128{}
	machineDense.One()
	machineDense.Zero()
	machineDense.Zero()
	machineDense.ControlledNot([]int{0}, 1)
	machineDense.ControlledNot([]int{0, 1}, 2)

	machineSparse := MachineSparse128{}
	machineSparse.One()
	machineSparse.Zero()
	machineSparse.Zero()
	machineSparse.ControlledNot([]int{0}, 1)
	machineSparse.ControlledNot([]int{0, 1}, 2)

	for i := 0; i < machineDense.R; i += machineDense.C {
		a := machineSparse.Matrix[i]
		for j := 0; j < machineDense.C; j++ {
			var value complex128
			if a != nil {
				value = a[j]
			}
			if value != machineDense.Matrix[i*machineDense.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, machineDense.Matrix[i*machineDense.C+j])
			}
		}
	}
}
