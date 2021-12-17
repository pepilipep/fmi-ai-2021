package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

type randomizedSet struct {
	m   map[int]int
	els []int
}

func (s *randomizedSet) Init() {
	s.m = make(map[int]int)
}

func (s *randomizedSet) Insert(el int) {
	if _, ok := s.m[el]; !ok {
		s.m[el] = len(s.els)
		s.els = append(s.els, el)
	}
}

func (s *randomizedSet) Len() int {
	return len(s.els)
}

func (s *randomizedSet) Delete(el int) {
	if idx, ok := s.m[el]; ok {
		s.els[idx] = s.els[len(s.els)-1]
		s.m[s.els[idx]] = idx
		s.els = s.els[:len(s.els)-1]

		delete(s.m, el)
	}
}

func (s *randomizedSet) Random() int {
	randIdx := rand.Intn(len(s.els))
	return s.els[randIdx]
}

type conflicts struct {
	n        int
	columns  map[int]map[int]bool
	primDiag map[int]map[int]bool
	secDiag  map[int]map[int]bool
}

func (c *conflicts) Init(n int) {
	c.n = n
	c.columns = make(map[int]map[int]bool)
	c.primDiag = make(map[int]map[int]bool)
	c.secDiag = make(map[int]map[int]bool)

	for row := 0; row < 2*c.n-1; row++ {
		c.primDiag[row] = make(map[int]bool)
		c.secDiag[row] = make(map[int]bool)
	}

	for row := 0; row < c.n; row++ {
		c.columns[row] = make(map[int]bool)
		c.Add(row, 0)
	}
}

func (c *conflicts) primIdx(row int, col int) int {
	return row + col
}

func (c *conflicts) primCol(row int, idx int) int {
	return idx - row
}

func (c *conflicts) secIdx(row int, col int) int {
	return c.n - 1 + col - row
}

func (c *conflicts) secCol(row int, idx int) int {
	return idx - c.n + 1 + row
}

func (c *conflicts) Get(row int, col int) int {
	return len(c.columns[col]) + len(c.primDiag[c.primIdx(row, col)]) + len(c.secDiag[c.secIdx(row, col)]) - 3
}

func (c *conflicts) Remove(row int, col int) []int {
	primIdx := c.primIdx(row, col)
	secIdx := c.secIdx(row, col)

	delete(c.columns[col], row)
	delete(c.primDiag[primIdx], row)
	delete(c.secDiag[secIdx], row)

	var nonConflicting []int
	if len(c.columns[col]) == 1 {
		for pot := range c.columns[col] {
			if c.Get(pot, col) == 0 {
				nonConflicting = append(nonConflicting, pot)
			}
		}
	}
	if len(c.primDiag[primIdx]) == 1 {
		for pot := range c.primDiag[primIdx] {
			if c.Get(pot, c.primCol(pot, primIdx)) == 0 {
				nonConflicting = append(nonConflicting, pot)
			}
		}
	}
	if len(c.secDiag[secIdx]) == 1 {
		for pot := range c.secDiag[secIdx] {
			if c.Get(pot, c.secCol(pot, secIdx)) == 0 {
				nonConflicting = append(nonConflicting, pot)
			}
		}
	}

	return nonConflicting
}

func (c *conflicts) Add(row int, col int) []int {
	primIdx := c.primIdx(row, col)
	secIdx := c.secIdx(row, col)

	var newConflicting []int
	if len(c.columns[col]) == 1 {
		for pot := range c.columns[col] {
			if c.Get(pot, col) == 0 {
				newConflicting = append(newConflicting, pot)
			}
		}
	}
	if len(c.primDiag[primIdx]) == 1 {
		for pot := range c.primDiag[primIdx] {
			if c.Get(pot, c.primCol(pot, primIdx)) == 0 {
				newConflicting = append(newConflicting, pot)
			}
		}
	}
	if len(c.secDiag[secIdx]) == 1 {
		for pot := range c.secDiag[secIdx] {
			if c.Get(pot, c.secCol(pot, secIdx)) == 0 {
				newConflicting = append(newConflicting, pot)
			}
		}
	}

	c.columns[col][row] = true
	c.primDiag[primIdx][row] = true
	c.secDiag[secIdx][row] = true

	return newConflicting
}

type MinConflicts struct {
	n           int
	queens      []int
	conflicting randomizedSet
	conflicts   conflicts
	shuffle     []int
}

func (m *MinConflicts) Read() {
	reader := bufio.NewReader(os.Stdin)

	line, err := reader.ReadString('\n')
	if err != nil {
		return
	}

	split := strings.Fields(line)
	if len(split) != 1 {
		return
	}

	m.n, _ = strconv.Atoi(split[0])

	m.conflicting.Init()

	m.queens = make([]int, m.n)
	for i := range m.queens {
		m.queens[i] = 0
		m.conflicting.Insert(i)
	}

	m.conflicts.Init(m.n)

	m.shuffle = rand.Perm(m.n)
}

func (m *MinConflicts) findMinColumn(row int) int {
	minCol, minConf := 0, m.n
	for _, col := range m.shuffle {
		if col != m.queens[row] {
			curr := m.conflicts.Get(row, col)
			if curr == -3 {
				return col
			}
			if curr < minConf {
				minConf = curr
				minCol = col
			}
		}
	}
	return minCol
}

func (m *MinConflicts) moveQueen(row int, newCol int) {
	nonConflicting := m.conflicts.Remove(row, m.queens[row])
	for _, nc := range nonConflicting {
		m.conflicting.Delete(nc)
	}
	m.queens[row] = newCol
	newConflicting := m.conflicts.Add(row, m.queens[row])
	for _, nc := range newConflicting {
		m.conflicting.Insert(nc)
	}

	if m.conflicts.Get(row, m.queens[row]) > 0 {
		m.conflicting.Insert(row)
	} else {
		m.conflicting.Delete(row)
	}
}

func (m *MinConflicts) Solve() {
	deleteme := 0
	for m.conflicting.Len() > 0 {
		deleteme++
		if deleteme%50 == 0 {
			m.shuffle = rand.Perm(m.n)
		}
		conf := m.conflicting.Random()

		newCol := m.findMinColumn(conf)

		m.moveQueen(conf, newCol)
	}
}

func (m *MinConflicts) Print() {
	for _, col := range m.queens {
		i := 0
		for ; i < col; i++ {
			fmt.Print("_ ")
		}
		fmt.Print("* ")
		for i++; i < m.n; i++ {
			fmt.Print("_ ")
		}
		fmt.Println()
	}
}

func main() {

	solver := MinConflicts{}

	solver.Read()

	startTime := time.Now()
	solver.Solve()

	dur := time.Since(startTime)
	fmt.Printf("%.3f\n", dur.Seconds())

	solver.Print()
}
