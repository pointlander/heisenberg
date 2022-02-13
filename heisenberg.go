// Copyright 2022 The Heisenberg Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heisenberg

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
)

// Qubit is a qubit
type Qubit uint64

// GateType is a type of gate
type GateType int

const (
	// GateTypeControlledNot controlled not gate
	GateTypeControlledNot GateType = iota
	// GateTypeI multiply by identity
	GateTypeI
	// GateTypeH multiply by Hadamard gate
	GateTypeH
	// GateTypeX multiply by Pauli X matrix
	GateTypeX
	// GateTypeY multiply by Pauli Y matrix
	GateTypeY
	// GateTypeZ multiply by Pauli Z matrix
	GateTypeZ
	// GateTypeS multiply by phase matrix
	GateTypeS
	// GateTypeT multiply by T matrix
	GateTypeT
	// GateTypeU multiply by U matrix
	GateTypeU
	// GateTypeRX rotate X gate
	GateTypeRX
	// GateTypeRY rotate Y gate
	GateTypeRY
	// GateTypeRZ rotate Z gate
	GateTypeRZ
)

// Gate is a gate
type Gate struct {
	GateType
	Qubits             []Qubit
	Target             Qubit
	Theta, Phi, Lambda float64
}

// Genome is a quantum circuit
type Genome struct {
	Gates         []Gate
	Fitness       float64
	Width         int
	Probabilities [][2][]float64
}

// Copy copies a genome
func (g *Genome) Copy() Genome {
	cp := Genome{}
	cp.Gates = make([]Gate, len(g.Gates))
	for i := range g.Gates {
		cp.Gates[i].GateType = g.Gates[i].GateType
		cp.Gates[i].Qubits = make([]Qubit, len(g.Gates[i].Qubits))
		copy(cp.Gates[i].Qubits, g.Gates[i].Qubits)
		cp.Gates[i].Target = g.Gates[i].Target
		cp.Gates[i].Theta = g.Gates[i].Theta
		cp.Gates[i].Phi = g.Gates[i].Phi
		cp.Gates[i].Lambda = g.Gates[i].Lambda
	}
	cp.Width = g.Width
	cp.Probabilities = g.Probabilities
	return cp
}

// Execute the gates
func (g *Genome) Execute() {
	fitness := 0.0
	for _, probability := range g.Probabilities {
		machine, qubits := MachineSparse64{}, []Qubit{}
		i := 0
		for i < len(probability[0]) {
			if probability[0][i] == 0 {
				qubits = append(qubits, machine.Zero())
			} else {
				qubits = append(qubits, machine.One())
			}
			i++
		}
		for i < g.Width {
			qubits = append(qubits, machine.Zero())
			i++
		}
		for _, gate := range g.Gates {
			switch gate.GateType {
			case GateTypeControlledNot:
				machine.ControlledNot(gate.Qubits, gate.Target)
			case GateTypeI:
				machine.I(gate.Qubits...)
			case GateTypeH:
				machine.H(gate.Qubits...)
			case GateTypeX:
				machine.X(gate.Qubits...)
			case GateTypeY:
				machine.Y(gate.Qubits...)
			case GateTypeZ:
				machine.Z(gate.Qubits...)
			case GateTypeS:
				machine.S(gate.Qubits...)
			case GateTypeT:
				machine.T(gate.Qubits...)
			case GateTypeU:
				machine.U(gate.Theta, gate.Phi, gate.Lambda, gate.Qubits...)
			case GateTypeRX:
				machine.RX(gate.Theta, gate.Qubits...)
			case GateTypeRY:
				machine.RY(gate.Theta, gate.Qubits...)
			case GateTypeRZ:
				machine.RZ(gate.Theta, gate.Qubits...)
			}
		}
		for i := 0; i < len(probability[1]); i++ {
			abs := cmplx.Abs(complex128(machine.Vector64[i]))
			x := abs*abs - probability[1][i]
			fitness += x * x
		}
	}
	g.Fitness = fitness
}

// Optimize is an implementation of genetic optimize
func Optimize(width, depth int, probabilities [][2][]float64) {
	rand.Seed(1)

	qubit := func(qubits []Qubit) Qubit {
		qubit := Qubit(0)
		for {
			qubit = Qubit(rand.Intn(width))
			contains := false
			for _, value := range qubits {
				if value == qubit {
					contains = true
				}
			}
			if !contains {
				break
			}
		}
		return qubit
	}

	qubits := func() []Qubit {
		qubits := make([]Qubit, 0, 8)
		q := rand.Intn(3)
		for k := 0; k < q; k++ {
			qubits = append(qubits, qubit(qubits))
		}
		return qubits
	}

	gate := func() Gate {
		gate := Gate{}
		n := rand.Intn(17)
		if n < 5 {
			gate.GateType = GateTypeControlledNot
			gate.Qubits = qubits()
			gate.Target = qubit([]Qubit{})
		} else if n < 7 {
			gate.GateType = GateTypeI
			gate.Qubits = qubits()
		} else if n < 8 {
			gate.GateType = GateTypeH
			gate.Qubits = qubits()
		} else if n < 9 {
			gate.GateType = GateTypeX
			gate.Qubits = qubits()
		} else if n < 10 {
			gate.GateType = GateTypeY
			gate.Qubits = qubits()
		} else if n < 11 {
			gate.GateType = GateTypeZ
			gate.Qubits = qubits()
		} else if n < 12 {
			gate.GateType = GateTypeS
			gate.Qubits = qubits()
		} else if n < 13 {
			gate.GateType = GateTypeT
			gate.Qubits = qubits()
		} else if n < 14 {
			gate.GateType = GateTypeU
			gate.Qubits = qubits()
			gate.Theta = 4 * math.Pi * rand.Float64()
			gate.Lambda = rand.Float64()
			gate.Phi = rand.Float64()
		} else if n < 15 {
			gate.GateType = GateTypeRX
			gate.Qubits = qubits()
			gate.Theta = 4 * math.Pi * rand.Float64()
		} else if n < 16 {
			gate.GateType = GateTypeRY
			gate.Qubits = qubits()
			gate.Theta = 4 * math.Pi * rand.Float64()
		} else if n < 17 {
			gate.GateType = GateTypeRZ
			gate.Qubits = qubits()
			gate.Theta = 4 * math.Pi * rand.Float64()
		}
		return gate
	}

	genomes := make([]Genome, 100)
	for i := 0; i < 100; i++ {
		gates := make([]Gate, 0, depth)
		for j := 0; j < depth; j++ {
			gates = append(gates, gate())
		}
		genomes[i].Gates = gates
		genomes[i].Width = width
		genomes[i].Probabilities = probabilities
	}

	for g := 0; g < 100; g++ {
		for i := range genomes {
			genomes[i].Execute()
		}
		sort.Slice(genomes, func(i, j int) bool {
			return genomes[i].Fitness < genomes[j].Fitness
		})
		genomes = genomes[:100]
		fmt.Println(genomes[0].Fitness)
		for range genomes[:10] {
			m1, m2 := rand.Intn(10), rand.Intn(10)
			c1, c2 := genomes[m1].Copy(), genomes[m2].Copy()
			g1, g2 := rand.Intn(depth), rand.Intn(depth)
			c1.Gates[g1], c2.Gates[g2] = c2.Gates[g2], c1.Gates[g1]
			genomes = append(genomes, c1, c2)
		}
		for i := range genomes {
			cp := genomes[i].Copy()
			g := rand.Intn(depth)
			cp.Gates[g] = gate()
		}
	}
}
