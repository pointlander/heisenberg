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

func TestTensor64(t *testing.T) {
	p := &Sparse64{
		R: 2,
		C: 2,
		Matrix: map[int]map[int]complex64{
			0: map[int]complex64{
				0: 1,
				1: 0,
			},
			1: map[int]complex64{
				0: 0,
				1: 1,
			},
		},
	}
	q := p.Copy()
	for i := 0; i < 3; i++ {
		q = p.Tensor(q)
	}

	m := &Matrix64{
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
		Matrix: map[int]map[int]complex128{
			0: map[int]complex128{
				0: 1,
				1: 0,
			},
			1: map[int]complex128{
				0: 0,
				1: 1,
			},
		},
	}
	q := p.Copy()
	for i := 0; i < 3; i++ {
		q = p.Tensor(q)
	}

	m := &Matrix128{
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
	a := Matrix64{}.ControlledNot(8, []int{0}, 1)
	b := Matrix64{}.ControlledNot(8, []int{0, 1}, 2)
	c := a.Multiply(b)

	d := Sparse64{}.ControlledNot(8, []int{0}, 1)
	e := Sparse64{}.ControlledNot(8, []int{0, 1}, 2)
	f := d.Multiply(e)

	for i := 0; i < c.R; i += c.C {
		a := f.Matrix[i]
		for j := 0; j < c.C; j++ {
			var value complex64
			if a != nil {
				value = a[j]
			}
			if value != c.Matrix[i*c.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, c.Matrix[i*c.C+j])
			}
		}
	}
}

func TestMultiply128(t *testing.T) {
	a := Matrix128{}.ControlledNot(8, []int{0}, 1)
	b := Matrix128{}.ControlledNot(8, []int{0, 1}, 2)
	c := a.Multiply(b)

	d := Sparse128{}.ControlledNot(8, []int{0}, 1)
	e := Sparse128{}.ControlledNot(8, []int{0, 1}, 2)
	f := d.Multiply(e)

	for i := 0; i < c.R; i += c.C {
		a := f.Matrix[i]
		for j := 0; j < c.C; j++ {
			var value complex128
			if a != nil {
				value = a[j]
			}
			if value != c.Matrix[i*c.C+j] {
				t.Fatalf("%d %d %f != %f", i, j, value, c.Matrix[i*c.C+j])
			}
		}
	}
}
