package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	GivenInput = iota
	RandomInput
)

const (
	OneSideCrossover = iota
	TwoSideCrossover
)

const (
	MutateSwap = iota
	MutateInsert
	MutateReverse
)

const (
	MAX_ITERATIONS  = 10000
	POPULATION_SIZE = 100
	MATING_SIZE     = 90
	MUTATION_PROB   = 0.5
	CROSSOVER_TYPE  = OneSideCrossover
	REPEAT_STOP     = 40
)

type coordinate struct {
	x float64
	y float64
}

func (c *coordinate) DistanceTo(other coordinate) float64 {
	difX, difY := c.x-other.x, c.y-other.y
	return math.Sqrt(difX*difX + difY*difY)
}

// chromosome is a permutation of N
type chromosome struct {
	genes   []int
	fitness float64
}

type TravellingSalesman struct {
	n           int
	points      []coordinate
	population  []chromosome
	bestHistory []chromosome
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

func (ts *TravellingSalesman) HandleInput(inputType int) {
	reader := bufio.NewReader(os.Stdin)

	line, err := reader.ReadString('\n')
	if err != nil {
		return
	}

	numbers, err := retrieveNumbers(line, 1)
	if err != nil {
		return
	}
	ts.n = numbers[0]

	switch inputType {
	case GivenInput:
		for i := 0; i < ts.n; i++ {
			line, err := reader.ReadString('\n')
			if err != nil {
				return
			}

			numbers, err = retrieveNumbers(line, 2)
			if err != nil {
				return
			}

			ts.points = append(ts.points, coordinate{x: float64(numbers[0]), y: float64(numbers[1])})
		}
	case RandomInput:
		for i := 0; i < ts.n; i++ {
			x, y := rand.Float64()*10000, rand.Float64()*10000
			ts.points = append(ts.points, coordinate{x, y})
		}
	}
}

func (ts *TravellingSalesman) initPopulation() {
	ts.population = make([]chromosome, POPULATION_SIZE)
	for i := 0; i < POPULATION_SIZE; i++ {
		ts.population[i].genes = rand.Perm(ts.n)
		ts.fitness(i)
	}
}

func (ts *TravellingSalesman) fitness(chrIdx int) {
	ts.population[chrIdx].fitness = 0
	for i := 0; i < ts.n-1; i++ {
		ts.population[chrIdx].fitness += ts.points[ts.population[chrIdx].genes[i]].DistanceTo(ts.points[ts.population[chrIdx].genes[i+1]])
	}
}

// Len implementation for sort.Interface
func (ts *TravellingSalesman) Len() int {
	return len(ts.population)
}

// Less implementation for sort.Interface
func (ts *TravellingSalesman) Less(i, j int) bool {
	return ts.population[i].fitness < ts.population[j].fitness
}

// Swap implementation for sort.Interface
func (ts *TravellingSalesman) Swap(i, j int) {
	tmp := ts.population[i]
	ts.population[i] = ts.population[j]
	ts.population[j] = tmp
}

func (ts *TravellingSalesman) calculateFitness() float64 {
	sum := float64(0)
	for i := range ts.population {
		ts.fitness(i)
		sum += ts.population[i].fitness
	}

	sort.Sort(ts)

	ts.bestHistory = append(ts.bestHistory, ts.population[0])

	return sum
}

func (ts *TravellingSalesman) selectMating(fitnessSum float64) []int {
	// begin with all in mating pool
	var matingPool []int
	for i := 0; i < POPULATION_SIZE; i++ {
		matingPool = append(matingPool, i)
	}

	minFit := ts.population[0].fitness * 2 / 3
	fitnessSum -= minFit * POPULATION_SIZE

	// choose one per step to remove from mating pool
	for i := 0; i < POPULATION_SIZE-MATING_SIZE; i++ {
		cut := rand.Float64() * fitnessSum
		currSum := float64(0)
		for idxInPool, j := range matingPool {
			currInd := ts.population[j]
			currSum += (currInd.fitness - minFit)

			if currSum >= cut {
				// remove from mating pool
				matingPool[idxInPool] = matingPool[len(matingPool)-1]
				matingPool = matingPool[:len(matingPool)-1]

				fitnessSum -= (currInd.fitness - minFit)
				break
			}
		}
	}

	if len(matingPool) != MATING_SIZE {
		fmt.Printf("this should be impossible, check out code: [%d]\n", len(matingPool))
	}

	sort.Ints(matingPool)

	return matingPool
}

func (ts *TravellingSalesman) oneSideCrossover(mother, father chromosome) chromosome {
	idx := rand.Intn(ts.n-2) + 1
	genes := make([]int, ts.n)

	seen := make(map[int]bool)

	for i, g := range mother.genes[:idx] {
		genes[i] = g
		seen[g] = true
	}

	for _, g := range father.genes {
		if _, ok := seen[g]; !ok {
			seen[g] = true
			genes[idx] = g
			idx++
		}
	}

	return chromosome{genes: genes}
}

func (ts *TravellingSalesman) twoSideCrossover(mother, father chromosome) chromosome {
	idx1 := rand.Intn(ts.n-2) + 1
	idx2 := rand.Intn(ts.n-2) + 1
	if idx2 < idx1 {
		idx1, idx2 = idx2, idx1
	}
	if idx1 == idx2 {
		idx2++
	}

	genes := make([]int, ts.n)

	seen := make(map[int]bool)

	for i := idx1; i < idx2; i++ {
		genes[i] = mother.genes[i]
		seen[mother.genes[i]] = true
	}

	idx := 0
	for _, g := range father.genes {
		if _, ok := seen[g]; !ok {
			if idx < idx2 && idx >= idx1 {
				idx = idx2
			}
			seen[g] = true
			genes[idx] = g
			idx++
		}
	}

	return chromosome{genes: genes}
}

func (ts *TravellingSalesman) crossover(matingPool []int) []chromosome {
	var newborn []chromosome
	for i := 0; i < len(matingPool); i += 2 {
		x1 := ts.population[matingPool[i]]
		x2 := ts.population[matingPool[i+1]]
		switch CROSSOVER_TYPE {
		case OneSideCrossover:
			newborn = append(newborn, ts.oneSideCrossover(x1, x2), ts.oneSideCrossover(x2, x1))
		case TwoSideCrossover:
			newborn = append(newborn, ts.twoSideCrossover(x1, x2), ts.oneSideCrossover(x2, x1))
		}
	}
	return newborn
}

func (ts *TravellingSalesman) combineGenerations(newborn []chromosome) {
	ts.population = ts.population[:POPULATION_SIZE-MATING_SIZE]
	ts.population = append(ts.population, newborn...)
}

func (ts *TravellingSalesman) mutate() {
	for i := POPULATION_SIZE - MATING_SIZE; i < POPULATION_SIZE; i++ {
		shouldMutate := rand.Float64()
		if shouldMutate <= MUTATION_PROB {
			switch rand.Intn(2) {
			case MutateSwap:
				idx1, idx2 := rand.Intn(ts.n), rand.Intn(ts.n)
				ts.population[i].genes[idx1], ts.population[i].genes[idx2] = ts.population[i].genes[idx2], ts.population[i].genes[idx1]
			case MutateInsert:
				idx1, idx2 := rand.Intn(ts.n), rand.Intn(ts.n)
				if idx2 < idx1 {
					idx1, idx2 = idx2, idx1
				}
				if idx1 == idx2 {
					if idx1 != 0 {
						idx1--
					} else {
						idx2++
					}
				}
				tmp := ts.population[i].genes[idx1]
				copy(ts.population[i].genes[idx1:idx2], ts.population[i].genes[idx1+1:idx2+1])
				ts.population[i].genes[idx2] = tmp
			}
		}
	}
}

func (ts *TravellingSalesman) printBest(iteration int) {
	fmt.Println("-------------------")
	fmt.Printf("ITERATION: %d\n", iteration)
	fmt.Printf("path sum: %.2f\n", ts.bestHistory[iteration].fitness)
	fmt.Printf("path: %v\n", ts.bestHistory[iteration].genes)
}

func (ts *TravellingSalesman) shouldContinue() bool {
	if len(ts.bestHistory) > MAX_ITERATIONS {
		return false
	}
	if len(ts.bestHistory) <= REPEAT_STOP {
		return true
	}
	for i := len(ts.bestHistory) - REPEAT_STOP; i < len(ts.bestHistory); i++ {
		if ts.bestHistory[i].fitness != ts.bestHistory[i-1].fitness {
			return true
		}
	}
	return false
}

func (ts *TravellingSalesman) PrintStatistics() {

	last := len(ts.bestHistory) - REPEAT_STOP
	if len(ts.bestHistory) > MAX_ITERATIONS {
		last = len(ts.bestHistory)
	}

	points := []int{
		0,
		last / 4,
		last / 2,
		3 * last / 4,
		last - 1,
	}
	for _, i := range points {
		ts.printBest(i)
	}
}

func (ts *TravellingSalesman) Solve() {
	ts.initPopulation()
	for iteration := 1; ts.shouldContinue(); iteration++ {
		totalFitness := ts.calculateFitness()
		matingPool := ts.selectMating(totalFitness)
		newborn := ts.crossover(matingPool)
		ts.combineGenerations(newborn)
		ts.mutate()
	}
}

func main() {

	rand.Seed(420)

	ts := TravellingSalesman{}
	ts.HandleInput(RandomInput)
	ts.Solve()
	ts.PrintStatistics()
}

// 12
// 0 0
// 383 0
// -27 -283
// 336 -270
// 69 -246
// 169 31
// 320 -160
// 180 -318
// 492 -131
// 112 -110
// 306 -108
// 217 -447
