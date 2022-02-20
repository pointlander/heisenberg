// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heisenberg

import (
	"math"
	"testing"
)

func round64(a complex64) complex64 {
	return complex(float32(int(real(a)*32))/32, float32(int(imag(a)*32))/32)
}

func round128(a complex128) complex128 {
	return complex(float64(int(real(a)*32))/32, float64(int(imag(a)*32))/32)
}

func BenchmarkDense64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineDense64{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]Qubit{0}, 1)
		machine.ControlledNot([]Qubit{0, 1}, 2)
	}
}

func BenchmarkDense128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineDense128{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]Qubit{0}, 1)
		machine.ControlledNot([]Qubit{0, 1}, 2)
	}
}

func BenchmarkSparse64(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineSparse64{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]Qubit{0}, 1)
		machine.ControlledNot([]Qubit{0, 1}, 2)
	}
}

func BenchmarkSparse128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineSparse128{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]Qubit{0}, 1)
		machine.ControlledNot([]Qubit{0, 1}, 2)
	}
}

func BenchmarkMatrix128(b *testing.B) {
	for n := 0; n < b.N; n++ {
		machine := MachineMatrix128{}
		for i := 0; i < 4; i++ {
			machine.One()
			machine.Zero()
		}
		machine.ControlledNot([]Qubit{0}, 1)
		machine.ControlledNot([]Qubit{0, 1}, 2)
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
	machineDense.ControlledNot([]Qubit{0}, 1)
	machineDense.ControlledNot([]Qubit{0, 1}, 2)

	machineSparse := MachineSparse64{}
	machineSparse.One()
	machineSparse.Zero()
	machineSparse.Zero()
	machineSparse.ControlledNot([]Qubit{0}, 1)
	machineSparse.ControlledNot([]Qubit{0, 1}, 2)

	a := machineSparse.Vector64
	for i := 0; i < machineDense.R; i++ {
		if a[i] != machineDense.Matrix[i] {
			t.Fatalf("%d %f != %f", i, a[i], machineDense.Matrix[i])
		}
	}
}

func TestMultiply128(t *testing.T) {
	machineDense := MachineDense128{}
	machineDense.One()
	machineDense.Zero()
	machineDense.Zero()
	machineDense.ControlledNot([]Qubit{0}, 1)
	machineDense.ControlledNot([]Qubit{0, 1}, 2)

	machineSparse := MachineSparse128{}
	machineSparse.One()
	machineSparse.Zero()
	machineSparse.Zero()
	machineSparse.ControlledNot([]Qubit{0}, 1)
	machineSparse.ControlledNot([]Qubit{0, 1}, 2)

	a := machineSparse.Vector128
	for i := 0; i < machineDense.R; i++ {
		if a[i] != machineDense.Matrix[i] {
			t.Fatalf("%d %f != %f", i, a[i], machineDense.Matrix[i])
		}
	}
}

func TestControlledNotSparse64(t *testing.T) {
	machine := MachineSparse64{}
	machine.One()
	machine.One()
	machine.Zero()

	a := machine.ControlledNot([]Qubit{0}, 2)
	b := machine.ControlledNot([]Qubit{0, 1}, 2)
	c := b.Multiply(a)
	d := c.Copy()
	d.Transpose()
	var sum complex64
	for i, ii := range c.Matrix {
		for j, jj := range ii {
			sum += d.Matrix[i][j] - jj
		}
	}
	if sum != 0 {
		t.Fatalf("sum %f should be zero", sum)
	}

	if machine.Vector64[0] != 0 ||
		machine.Vector64[1] != 0 ||
		machine.Vector64[2] != 0 ||
		machine.Vector64[3] != 0 ||
		machine.Vector64[4] != 0 ||
		machine.Vector64[5] != 0 ||
		machine.Vector64[6] != 1 ||
		machine.Vector64[7] != 0 {
		t.Fatalf("machine has bad state %f", machine.Vector64)
	}
}

func TestControlledNotSparse128(t *testing.T) {
	machine := MachineSparse128{}
	machine.One()
	machine.One()
	machine.Zero()

	a := machine.ControlledNot([]Qubit{0}, 2)
	b := machine.ControlledNot([]Qubit{0, 1}, 2)
	c := b.Multiply(a)
	d := c.Copy()
	d.Transpose()
	var sum complex128
	for i, ii := range c.Matrix {
		for j, jj := range ii {
			sum += d.Matrix[i][j] - jj
		}
	}
	if sum != 0 {
		t.Fatalf("sum %f should be zero", sum)
	}

	if machine.Vector128[0] != 0 ||
		machine.Vector128[1] != 0 ||
		machine.Vector128[2] != 0 ||
		machine.Vector128[3] != 0 ||
		machine.Vector128[4] != 0 ||
		machine.Vector128[5] != 0 ||
		machine.Vector128[6] != 1 ||
		machine.Vector128[7] != 0 {
		t.Fatalf("machine has bad state %f", machine.Vector128)
	}
}

func TestControlledNotDense64(t *testing.T) {
	machine := MachineDense64{}
	machine.One()
	machine.One()
	machine.Zero()

	a := machine.ControlledNot([]Qubit{0}, 2)
	b := machine.ControlledNot([]Qubit{0, 1}, 2)
	c := b.Multiply(a)
	d := c.Copy()
	d.Transpose()
	var sum complex64
	for i := range c.Matrix {
		sum += d.Matrix[i] - c.Matrix[i]
	}
	if sum != 0 {
		t.Fatalf("sum %f should be zero", sum)
	}

	if machine.Dense64.Matrix[0] != 0 ||
		machine.Dense64.Matrix[1] != 0 ||
		machine.Dense64.Matrix[2] != 0 ||
		machine.Dense64.Matrix[3] != 0 ||
		machine.Dense64.Matrix[4] != 0 ||
		machine.Dense64.Matrix[5] != 0 ||
		machine.Dense64.Matrix[6] != 1 ||
		machine.Dense64.Matrix[7] != 0 {
		t.Fatalf("machine has bad state %f", machine.Dense64.Matrix)
	}
}

func TestControlledNotDense128(t *testing.T) {
	machine := MachineDense128{}
	machine.One()
	machine.One()
	machine.Zero()

	a := machine.ControlledNot([]Qubit{0}, 2)
	b := machine.ControlledNot([]Qubit{0, 1}, 2)
	c := b.Multiply(a)
	d := c.Copy()
	d.Transpose()
	var sum complex128
	for i := range c.Matrix {
		sum += d.Matrix[i] - c.Matrix[i]
	}
	if sum != 0 {
		t.Fatalf("sum %f should be zero", sum)
	}

	if machine.Dense128.Matrix[0] != 0 ||
		machine.Dense128.Matrix[1] != 0 ||
		machine.Dense128.Matrix[2] != 0 ||
		machine.Dense128.Matrix[3] != 0 ||
		machine.Dense128.Matrix[4] != 0 ||
		machine.Dense128.Matrix[5] != 0 ||
		machine.Dense128.Matrix[6] != 1 ||
		machine.Dense128.Matrix[7] != 0 {
		t.Fatalf("machine has bad state %f", machine.Dense128.Matrix)
	}
}

func TestRXSparse64(t *testing.T) {
	machine := MachineSparse64{}
	qubit := machine.Zero()
	machine.RX(4*math.Pi, qubit)
	if round64(machine.Vector64[0]) != 1 && round64(machine.Vector64[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRYSparse64(t *testing.T) {
	machine := MachineSparse64{}
	qubit := machine.Zero()
	machine.RY(4*math.Pi, qubit)
	if round64(machine.Vector64[0]) != 1 && round64(machine.Vector64[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRZSparse64(t *testing.T) {
	machine := MachineSparse64{}
	qubit := machine.Zero()
	machine.RZ(4*math.Pi, qubit)
	if round64(machine.Vector64[0]) != 1 && round64(machine.Vector64[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRXSparse128(t *testing.T) {
	machine := MachineSparse128{}
	qubit := machine.Zero()
	machine.RX(4*math.Pi, qubit)
	if round128(machine.Vector128[0]) != 1 && round128(machine.Vector128[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRYSparse128(t *testing.T) {
	machine := MachineSparse128{}
	qubit := machine.Zero()
	machine.RY(4*math.Pi, qubit)
	if round128(machine.Vector128[0]) != 1 && round128(machine.Vector128[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRZSparse128(t *testing.T) {
	machine := MachineSparse128{}
	qubit := machine.Zero()
	machine.RZ(4*math.Pi, qubit)
	if round128(machine.Vector128[0]) != 1 && round128(machine.Vector128[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRXDense64(t *testing.T) {
	machine := MachineDense64{}
	qubit := machine.Zero()
	machine.RX(4*math.Pi, qubit)
	if round64(machine.Matrix[0]) != 1 && round64(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRYDense64(t *testing.T) {
	machine := MachineDense64{}
	qubit := machine.Zero()
	machine.RY(4*math.Pi, qubit)
	if round64(machine.Matrix[0]) != 1 && round64(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRZDense64(t *testing.T) {
	machine := MachineDense64{}
	qubit := machine.Zero()
	machine.RZ(4*math.Pi, qubit)
	if round64(machine.Matrix[0]) != 1 && round64(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRXDense128(t *testing.T) {
	machine := MachineDense128{}
	qubit := machine.Zero()
	machine.RX(4*math.Pi, qubit)
	if round128(machine.Matrix[0]) != 1 && round128(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRYDense128(t *testing.T) {
	machine := MachineDense128{}
	qubit := machine.Zero()
	machine.RY(4*math.Pi, qubit)
	if round128(machine.Matrix[0]) != 1 && round128(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestRZDense128(t *testing.T) {
	machine := MachineDense128{}
	qubit := machine.Zero()
	machine.RZ(4*math.Pi, qubit)
	if round128(machine.Matrix[0]) != 1 && round128(machine.Matrix[1]) != 0 {
		t.Fatal("invalid qubit value")
	}
}

func TestSwapDense64(t *testing.T) {
	machine := MachineDense64{}
	q0 := machine.Zero()
	q1 := machine.One()
	machine.Swap(q0, q1)
	expect := []complex64{0, 0, 1, 0}
	for i, value := range expect {
		if value != machine.Matrix[i] {
			t.Fatal("swap didn't work")
		}
	}
}

func TestSwapDense128(t *testing.T) {
	machine := MachineDense128{}
	q0 := machine.Zero()
	q1 := machine.One()
	machine.Swap(q0, q1)
	expect := []complex128{0, 0, 1, 0}
	for i, value := range expect {
		if value != machine.Matrix[i] {
			t.Fatal("swap didn't work")
		}
	}
}

func TestSwapSparse64(t *testing.T) {
	machine := MachineSparse64{}
	q0 := machine.Zero()
	q1 := machine.One()
	machine.Swap(q0, q1)
	expect := []complex64{0, 0, 1, 0}
	for i, value := range expect {
		if value != machine.Vector64[i] {
			t.Fatal("swap didn't work")
		}
	}
}

func TestSwapSparse128(t *testing.T) {
	machine := MachineSparse128{}
	q0 := machine.Zero()
	q1 := machine.One()
	machine.Swap(q0, q1)
	expect := []complex128{0, 0, 1, 0}
	for i, value := range expect {
		if value != machine.Vector128[i] {
			t.Fatal("swap didn't work")
		}
	}
}
