package main

import (
	"bufio"
	"container/heap"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

type operation string

const (
	operationLeft  = operation("left")
	operationRight = operation("right")
	operationUp    = operation("up")
	operationDown  = operation("down")
)

type coordinate struct {
	x int
	y int
}

type solution struct {
	Operations []operation
}

func (s *solution) Print() {
	fmt.Println(len(s.Operations))
	for _, op := range s.Operations {
		fmt.Println(op)
	}
}

func (c *coordinate) toIndex(m int) int {
	return c.x*m + c.y
}

func coordinateFromIndex(m int, idx int) coordinate {
	return coordinate{
		x: idx / m,
		y: idx % m,
	}
}

type PuzzleSolution struct {
	n                    int
	m                    int
	zeroIndex            coordinate
	table                [][]int
	currentZero          coordinate
	precomputedManhattan *int
	path                 []operation
}

func retrieveNumbers(line string, expectedCount int) ([]int, error) {
	sep := strings.Fields(line)
	if len(sep) != expectedCount {
		return nil, fmt.Errorf("expected number count: [%d], found count: [%d]", expectedCount, len(sep))
	}

	nums := make([]int, len(sep))
	for i, s := range sep {
		num, err := strconv.Atoi(s)
		if err != nil {
			return nil, err
		}
		nums[i] = num
	}

	return nums, nil
}

func (p *PuzzleSolution) Read() error {
	reader := bufio.NewReader(os.Stdin)

	line, err := reader.ReadString('\n')
	if err != nil {
		return err
	}
	numbers, err := retrieveNumbers(line, 1)
	if err != nil {
		return err
	}
	p.n = numbers[0]
	p.m = int(math.Sqrt(float64(p.n + 1)))

	line, err = reader.ReadString('\n')
	if err != nil {
		return err
	}
	numbers, err = retrieveNumbers(line, 1)
	if err != nil {
		return err
	}
	if numbers[0] != -1 {
		p.zeroIndex = coordinateFromIndex(p.m, numbers[0]-1)
	} else {
		p.zeroIndex = coordinateFromIndex(p.m, p.n)
	}

	p.table = make([][]int, p.m)

	for row := 0; row < p.m; row++ {
		line, err = reader.ReadString('\n')
		if err != nil {
			return err
		}
		numbers, err = retrieveNumbers(line, p.m)
		if err != nil {
			return err
		}
		p.table[row] = make([]int, p.m)
		p.table[row] = numbers
		for idx, num := range numbers {
			if num == 0 {
				p.currentZero = coordinate{x: row, y: idx}
			}
		}
	}

	return nil
}

func (p *PuzzleSolution) inversionsCount() int {
	counter := 0

	for i_right := 0; i_right < p.m; i_right++ {
		for j_right := 0; j_right < p.m; j_right++ {
			if p.table[i_right][j_right] == 0 {
				continue
			}
			for i_left := 0; i_left <= i_right; i_left++ {
				for j_left := 0; j_left < p.m; j_left++ {
					if i_left == i_right && j_left == j_right {
						break
					}
					if p.table[i_left][j_left] == 0 {
						continue
					}
					if p.table[i_left][j_left] > p.table[i_right][j_right] {
						counter++
					}
				}
			}
		}
	}

	return counter
}

func (p *PuzzleSolution) IsSolvable() bool {
	invCount := p.inversionsCount()

	if p.m%2 == 1 {
		return (invCount+p.zeroIndex.x)%2 == 1
	}

	return (invCount+p.currentZero.x+p.zeroIndex.x)%2 == 0
}

func manhattan(p1, p2 coordinate) int {
	dist := 0
	if p1.x > p2.x {
		dist += p1.x - p2.x
	} else {
		dist += p2.x - p1.x
	}
	if p1.y > p2.y {
		dist += p1.y - p2.y
	} else {
		dist += p2.y - p1.y
	}
	return dist
}

func (p *PuzzleSolution) tileShouldBe(idx int) coordinate {
	if p.zeroIndex.toIndex(p.m) < idx {
		return coordinateFromIndex(p.m, idx)
	}
	return coordinateFromIndex(p.m, idx-1)
}

func (p *PuzzleSolution) manhattan() int {
	if p.precomputedManhattan != nil {
		return *p.precomputedManhattan
	}

	dist := 0

	for i := 0; i < p.m; i++ {
		for j := 0; j < p.m; j++ {
			current := coordinate{x: i, y: j}
			if p.table[i][j] != 0 {
				shouldBe := p.tileShouldBe(p.table[i][j])
				dist += manhattan(shouldBe, current)
			}
		}
	}

	p.precomputedManhattan = &dist

	return dist
}

func (p *PuzzleSolution) priority() int {
	return p.manhattan() + len(p.path)
}

func copyTable(src, dst [][]int, idx1, idx2 int) {
	for i := range src {
		if i == idx1 || i == idx2 {
			dst[i] = make([]int, len(src[i]))
			copy(dst[i], src[i])
		} else {
			dst[i] = src[i]
		}
	}
}

func (p *PuzzleSolution) move(op operation) (*PuzzleSolution, bool) {

	newPuzzle := PuzzleSolution{
		n:           p.n,
		m:           p.m,
		zeroIndex:   p.zeroIndex,
		table:       make([][]int, p.m),
		currentZero: p.currentZero,
	}

	newPuzzle.path = make([]operation, len(p.path))
	copy(newPuzzle.path, p.path)
	newPuzzle.path = append(newPuzzle.path, op)

	switch op {
	case operationUp:
		if p.currentZero.x+1 == p.m {
			return nil, false
		}
		if len(p.path) > 0 && p.path[len(p.path)-1] == operationDown {
			return nil, false
		}
		copyTable(p.table, newPuzzle.table, p.currentZero.x, p.currentZero.x+1)
		newPuzzle.table[p.currentZero.x][p.currentZero.y] = p.table[p.currentZero.x+1][p.currentZero.y]
		newPuzzle.table[p.currentZero.x+1][p.currentZero.y] = p.table[p.currentZero.x][p.currentZero.y]
		newPuzzle.currentZero.x++

	case operationRight:
		if p.currentZero.y == 0 {
			return nil, false
		}
		if len(p.path) > 0 && p.path[len(p.path)-1] == operationLeft {
			return nil, false
		}
		copyTable(p.table, newPuzzle.table, p.currentZero.x, -1)
		newPuzzle.table[p.currentZero.x][p.currentZero.y] = p.table[p.currentZero.x][p.currentZero.y-1]
		newPuzzle.table[p.currentZero.x][p.currentZero.y-1] = p.table[p.currentZero.x][p.currentZero.y]
		newPuzzle.currentZero.y--

	case operationLeft:
		if p.currentZero.y+1 == p.m {
			return nil, false
		}
		if len(p.path) > 0 && p.path[len(p.path)-1] == operationRight {
			return nil, false
		}
		copyTable(p.table, newPuzzle.table, p.currentZero.x, -1)
		newPuzzle.table[p.currentZero.x][p.currentZero.y] = p.table[p.currentZero.x][p.currentZero.y+1]
		newPuzzle.table[p.currentZero.x][p.currentZero.y+1] = p.table[p.currentZero.x][p.currentZero.y]
		newPuzzle.currentZero.y++

	case operationDown:
		if p.currentZero.x == 0 {
			return nil, false
		}
		if len(p.path) > 0 && p.path[len(p.path)-1] == operationUp {
			return nil, false
		}
		copyTable(p.table, newPuzzle.table, p.currentZero.x, p.currentZero.x-1)
		newPuzzle.table[p.currentZero.x][p.currentZero.y] = p.table[p.currentZero.x-1][p.currentZero.y]
		newPuzzle.table[p.currentZero.x-1][p.currentZero.y] = p.table[p.currentZero.x][p.currentZero.y]
		newPuzzle.currentZero.x--
	}

	destPlace := p.tileShouldBe(p.table[newPuzzle.currentZero.x][newPuzzle.currentZero.y])

	newMan := *p.precomputedManhattan -
		manhattan(newPuzzle.currentZero, destPlace) +
		manhattan(p.currentZero, destPlace)

	newPuzzle.precomputedManhattan = &newMan

	return &newPuzzle, true
}

func (p *PuzzleSolution) Neighbors() []*PuzzleSolution {
	var neighs []*PuzzleSolution
	for _, op := range []operation{operationDown, operationLeft, operationRight, operationUp} {
		neigh, ok := p.move(op)
		if ok {
			neighs = append(neighs, neigh)
		}
	}
	return neighs
}

func (p *PuzzleSolution) IsGoal() bool {
	return p.manhattan() == 0
}

func (p *PuzzleSolution) toSolution() *solution {
	return &solution{
		Operations: p.path,
	}
}

func (p *PuzzleSolution) solveWithCutOff(cutOff int) (*solution, bool) {
	pq := &PuzzleSolutionArray{}
	heap.Init(pq)
	heap.Push(pq, p)

	for pq.Len() > 0 {
		curr := heap.Pop(pq).(*PuzzleSolution)
		if curr.IsGoal() {
			return curr.toSolution(), true
		}
		neighs := curr.Neighbors()
		for _, nb := range neighs {
			if nb.priority() <= cutOff {
				heap.Push(pq, nb)
			}
		}
	}
	return nil, false
}

func (p *PuzzleSolution) Solve() {
	startTime := time.Now()
	startMan := p.manhattan()
	for cutOff := startMan; ; cutOff++ {
		pot, ok := p.solveWithCutOff(cutOff)
		if ok {
			dur := time.Since(startTime)
			fmt.Printf("%.3f\n", dur.Seconds())
			pot.Print()
			return
		}
	}
}

type PuzzleSolutionArray []*PuzzleSolution

func (p PuzzleSolutionArray) Len() int {
	return len(p)
}

func (p PuzzleSolutionArray) Less(i, j int) bool {
	return p[i].priority() < p[j].priority()
}

func (p PuzzleSolutionArray) Swap(i, j int) {
	tmp := p[i]
	p[i] = p[j]
	p[j] = tmp
}

func (p *PuzzleSolutionArray) Push(x interface{}) {
	*p = append(*p, x.(*PuzzleSolution))
}

func (p *PuzzleSolutionArray) Pop() interface{} {
	old := *p
	n := len(old)
	x := old[n-1]
	*p = old[0 : n-1]
	return x
}

func main() {

	puzzleSolution := PuzzleSolution{}

	if err := puzzleSolution.Read(); err != nil {
		fmt.Printf("error found: [%v]", err)
		os.Exit(1)
	}

	if !puzzleSolution.IsSolvable() {
		fmt.Println("puzzle is not solvable...")
		os.Exit(1)
	}

	puzzleSolution.Solve()
}
