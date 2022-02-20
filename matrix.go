// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heisenberg

import (
	"fmt"
	"math"
	"math/cmplx"
)

const (
	// CutoffPercent is the precent for sparse matrix cutoff
	CutoffPercent = .1
	// CutoffSize is the size of a matrix row required for dense
	CutoffSize = 256
)

// Matrix128 is an algebriac matrix
type Matrix128 struct {
	R, C   int
	Matrix []interface{}
}

func (a Matrix128) String() string {
	output := ""
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			switch row := a.Matrix[i].(type) {
			case []complex128:
				output += fmt.Sprintf("%f ", row[j])
			case map[int]complex128:
				output += fmt.Sprintf("%f ", row[j])
			default:
				output += fmt.Sprintf("0 ")
			}
		}
		output += fmt.Sprintf("\n")
	}
	return output
}

// Set set a value
func (a *Matrix128) Set(i, j int, value complex128) {
	if value == 0 {
		return
	}
	switch row := a.Matrix[i].(type) {
	case []complex128:
		row[j] = value
	case map[int]complex128:
		row[j] = value
		if length := len(row); float64(length)/float64(a.C) > CutoffPercent &&
			length > CutoffSize {
			r := make([]complex128, a.C)
			for key, value := range row {
				r[key] = value
			}
			a.Matrix[i] = r
		}
	default:
		r := make(map[int]complex128)
		r[j] = value
		a.Matrix[i] = r
	}
}

// Get get a value
func (a *Matrix128) Get(i, j int) complex128 {
	switch row := a.Matrix[i].(type) {
	case []complex128:
		return row[j]
	case map[int]complex128:
		return row[j]
	default:
		return 0
	}
}

// MachineMatrix128 is a 128 bit sparse matrix machine
type MachineMatrix128 struct {
	Vector128
	Qubits int
}

// Zero adds a zero to the matrix
func (a *MachineMatrix128) Zero() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	zero := Vector128{1, 0}
	if qubit == 0 {
		a.Vector128 = zero
		return qubit
	}
	a.Vector128 = a.Tensor(zero)
	return qubit
}

// One adds a one to the matrix
func (a *MachineMatrix128) One() Qubit {
	qubit := Qubit(a.Qubits)
	a.Qubits++
	one := Vector128{0, 1}
	if qubit == 0 {
		a.Vector128 = one
		return qubit
	}
	a.Vector128 = a.Tensor(one)
	return qubit
}

// Tensor product is the tensor product
func (a *Matrix128) Tensor(b *Matrix128) *Matrix128 {
	output := make([]interface{}, a.R*b.R)
	width := a.C * b.C
	for x, xx := range a.Matrix {
		for y, yy := range b.Matrix {
			switch row := output[x*b.R+y].(type) {
			case []complex128:
				switch rowA := xx.(type) {
				case []complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row[i*b.C+j] = ii * jj
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row[i*b.C+j] = ii * jj
							}
						}
					default:
					}
				case map[int]complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row[i*b.C+j] = ii * jj
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row[i*b.C+j] = ii * jj
							}
						}
					default:
					}
				default:
				}
			case map[int]complex128:
				switch rowA := xx.(type) {
				case []complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								if length := len(row); length > 0 {
									if float64(length)/float64(width) > CutoffPercent &&
										length > CutoffSize {
										v := make([]complex128, width)
										for key, value := range row {
											v[key] = value
										}
										output[x*b.R+y] = v
									} else {
										output[x*b.R+y] = row
									}
								}
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								if length := len(row); length > 0 {
									if float64(length)/float64(width) > CutoffPercent &&
										length > CutoffSize {
										v := make([]complex128, width)
										for key, value := range row {
											v[key] = value
										}
										output[x*b.R+y] = v
									} else {
										output[x*b.R+y] = row
									}
								}
							}
						}
					default:
					}
				case map[int]complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								if length := len(row); length > 0 {
									if float64(length)/float64(width) > CutoffPercent &&
										length > CutoffSize {
										v := make([]complex128, width)
										for key, value := range row {
											v[key] = value
										}
										output[x*b.R+y] = v
									} else {
										output[x*b.R+y] = row
									}
								}
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								if length := len(row); length > 0 {
									if float64(length)/float64(width) > CutoffPercent &&
										length > CutoffSize {
										v := make([]complex128, width)
										for key, value := range row {
											v[key] = value
										}
										output[x*b.R+y] = v
									} else {
										output[x*b.R+y] = row
									}
								}
							}
						}
					default:
					}
				default:
				}
			default:
				switch rowA := xx.(type) {
				case []complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row := make(map[int]complex128)
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								output[x*b.R+y] = row
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row := make(map[int]complex128)
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								output[x*b.R+y] = row
							}
						}
					default:
					}
				case map[int]complex128:
					switch rowB := yy.(type) {
					case []complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row := make(map[int]complex128)
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								output[x*b.R+y] = row
							}
						}
					case map[int]complex128:
						for i, ii := range rowA {
							for j, jj := range rowB {
								row := make(map[int]complex128)
								value := ii * jj
								if value != 0 {
									row[i*b.C+j] = value
								}
								output[x*b.R+y] = row
							}
						}
					default:
					}
				default:
				}
			}
		}
	}

	return &Matrix128{
		R:      a.R * b.R,
		C:      a.C * b.C,
		Matrix: output,
	}
}

// Multiply multiplies to matricies
func (a *Matrix128) Multiply(b *Matrix128) *Matrix128 {
	if a.C != b.R {
		panic("invalid dimensions")
	}
	output := Matrix128{
		R:      a.R,
		C:      b.C,
		Matrix: make([]interface{}, a.R),
	}
	for j := 0; j < b.C; j++ {
		for x, xx := range a.Matrix {
			switch rowA := xx.(type) {
			case []complex128:
				var sum complex128
				for y, value := range rowA {
					sum += b.Get(j, y) * value
				}
				output.Set(j, x, sum)
			case map[int]complex128:
				var sum complex128
				for y, value := range rowA {
					sum += b.Get(j, y) * value
				}
				output.Set(j, x, sum)
			default:
			}
		}
	}
	return &output
}

// Transpose transposes a matrix
func (a *Matrix128) Transpose() *Matrix128 {
	b := Matrix128{
		R:      a.C,
		C:      a.R,
		Matrix: make([]interface{}, a.R),
	}
	for i := 0; i < a.R; i++ {
		for j := 0; j < a.C; j++ {
			b.Set(j, i, a.Get(i, j))
		}
	}
	return &b
}

// Copy copies a matrix`
func (a *Matrix128) Copy() *Matrix128 {
	cp := &Matrix128{
		R:      a.R,
		C:      a.C,
		Matrix: make([]interface{}, a.R),
	}
	for a, aa := range a.Matrix {
		switch row := aa.(type) {
		case []complex128:
			value := make([]complex128, 0, len(row))
			for _, bb := range row {
				value = append(value, bb)
			}
			cp.Matrix[a] = value
		case map[int]complex128:
			value := make(map[int]complex128)
			for b, bb := range row {
				value[b] = bb
			}
			cp.Matrix[a] = value
		default:
		}
	}
	return cp
}

// MultiplyVector multiplies a matrix by a vector
func (a *Matrix128) MultiplyVector(b Vector128) Vector128 {
	if a.C != len(b) {
		panic(fmt.Sprintf("invalid dimensions %d %d", a.C, len(b)))
	}
	output := make(Vector128, 0, a.R)
	for _, xx := range a.Matrix {
		switch row := xx.(type) {
		case []complex128:
			var sum complex128
			for y, value := range row {
				sum += b[y] * value
			}
			output = append(output, sum)
		case map[int]complex128:
			var sum complex128
			for y, value := range row {
				sum += b[y] * value
			}
			output = append(output, sum)
		default:
			output = append(output, 0)
		}
	}
	return output
}

// ControlledNot controlled not gate
func (a *MachineMatrix128) ControlledNot(c []Qubit, t Qubit) *Matrix128 {
	n := a.Qubits
	p := &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1,
			},
		},
	}
	q := p
	for i := 0; i < n-1; i++ {
		q = p.Tensor(q)
	}
	d := q.R

	index := make([]int64, 0)
	for i := 0; i < d; i++ {
		bits := int64(i)

		// Apply X
		apply := true
		for _, j := range c {
			if (bits>>(Qubit(n-1)-j))&1 == 0 {
				apply = false
				break
			}
		}

		if apply {
			if (bits>>(Qubit(n-1)-t))&1 == 0 {
				bits |= 1 << (Qubit(n-1) - t)
			} else {
				bits &= ^(1 << (Qubit(n-1) - t))
			}
		}

		index = append(index, bits)
	}

	g := Matrix128{
		R:      q.R,
		C:      q.C,
		Matrix: make([]interface{}, q.R),
	}
	for i, ii := range index {
		g.Matrix[i] = q.Matrix[int(ii)]
	}

	a.Vector128 = g.MultiplyVector(a.Vector128)

	return &g
}

// Multiply multiplies the machine by a matrix
func (a *MachineMatrix128) Multiply(b *Matrix128, qubits ...Qubit) {
	indexes := make(map[int]bool)
	for _, value := range qubits {
		indexes[int(value)] = true
	}

	identity := IMatrix128()
	d := IMatrix128()
	if indexes[0] {
		d = b.Copy()
	}
	for i := 1; i < a.Qubits; i++ {
		if indexes[i] {
			d = d.Tensor(b)
			continue
		}

		d = d.Tensor(identity)
	}

	a.Vector128 = d.MultiplyVector(a.Vector128)
}

// IMatrix128 identity matrix
func IMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1,
			},
		},
	}
}

// I multiply by identity
func (a *MachineMatrix128) I(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(IMatrix128(), qubits...)
	return a
}

// HMatrix128 Hadamard matrix
func HMatrix128() *Matrix128 {
	v := complex(1/math.Sqrt2, 0)
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: v,
				1: v,
			},
			map[int]complex128{
				0: v,
				1: -v,
			},
		},
	}
}

// H multiply by Hadamard gate
func (a *MachineMatrix128) H(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(HMatrix128(), qubits...)
	return a
}

// XMatrix128 Pauli X matrix
func XMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				1: 1,
			},
			map[int]complex128{
				0: 1,
			},
		},
	}
}

// X multiply by Pauli X matrix
func (a *MachineMatrix128) X(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(XMatrix128(), qubits...)
	return a
}

// YMatrix128 Pauli Y matrix
func YMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				1: -1i,
			},
			map[int]complex128{
				0: 1i,
			},
		},
	}
}

// Y multiply by Pauli Y matrix
func (a *MachineMatrix128) Y(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(YMatrix128(), qubits...)
	return a
}

// ZMatrix128 Pauli Z matrix
func ZMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: -1,
			},
		},
	}
}

// Z multiply by Pauli Z matrix
func (a *MachineMatrix128) Z(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(ZMatrix128(), qubits...)
	return a
}

// SMatrix128 phase gate
func SMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: 1i,
			},
		},
	}
}

// S multiply by phase matrix
func (a *MachineMatrix128) S(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(SMatrix128(), qubits...)
	return a
}

// TMatrix128 T gate
func TMatrix128() *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: 1,
			},
			map[int]complex128{
				1: cmplx.Exp(1i * math.Pi / 4),
			},
		},
	}
}

// T multiply by T matrix
func (a *MachineMatrix128) T(qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(TMatrix128(), qubits...)
	return a
}

// UMatrix128 U gate
func UMatrix128(theta, phi, lambda float64) *Matrix128 {
	v := complex(theta/2, 0)
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: cmplx.Cos(v),
				1: -1 * cmplx.Exp(complex(0, lambda)) * cmplx.Sin(v),
			},
			map[int]complex128{
				0: cmplx.Exp(complex(0, phi)) * cmplx.Sin(v),
				1: cmplx.Exp(complex(0, (phi+lambda))) * cmplx.Cos(v),
			},
		},
	}
}

// U multiply by U matrix
func (a *MachineMatrix128) U(theta, phi, lambda float64, qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(UMatrix128(theta, phi, lambda), qubits...)
	return a
}

// RXMatrix128 x rotation matrix
func RXMatrix128(theta complex128) *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: cmplx.Cos(theta),
				1: -1i * cmplx.Sin(theta),
			},
			map[int]complex128{
				0: -1i * cmplx.Sin(theta),
				1: cmplx.Cos(theta),
			},
		},
	}
}

// RX rotate X gate
func (a *MachineMatrix128) RX(theta float64, qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(RXMatrix128(complex(theta/2, 0)), qubits...)
	return a
}

// RYMatrix128 y rotation matrix
func RYMatrix128(theta complex128) *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: cmplx.Cos(theta),
				1: -1 * cmplx.Sin(theta),
			},
			map[int]complex128{
				0: cmplx.Sin(theta),
				1: cmplx.Cos(theta),
			},
		},
	}
}

// RY rotate Y gate
func (a *MachineMatrix128) RY(theta float64, qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(RYMatrix128(complex(theta/2, 0)), qubits...)
	return a
}

// RZMatrix128 z rotation matrix
func RZMatrix128(theta complex128) *Matrix128 {
	return &Matrix128{
		R: 2,
		C: 2,
		Matrix: []interface{}{
			map[int]complex128{
				0: cmplx.Exp(-1 * theta),
			},
			map[int]complex128{
				1: cmplx.Exp(theta),
			},
		},
	}
}

// RZ rotate Z gate
func (a *MachineMatrix128) RZ(theta float64, qubits ...Qubit) *MachineMatrix128 {
	a.Multiply(RZMatrix128(complex(theta/2, 0)), qubits...)
	return a
}

// Swap swaps qubits`
func (a *MachineMatrix128) Swap(qubits ...Qubit) *MachineMatrix128 {
	length := len(qubits)

	for i := 0; i < length/2; i++ {
		c, t := qubits[i], qubits[(length-1)-i]
		a.ControlledNot([]Qubit{c}, t)
		a.ControlledNot([]Qubit{t}, c)
		a.ControlledNot([]Qubit{c}, t)
	}

	return a
}
