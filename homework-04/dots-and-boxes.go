package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	minTime = time.Second * 20
)

const (
	computerPlayer = iota
	personPlayer
)

const (
	verticalLine = iota
	horizontalLine
)

var playerToSymbol = map[int]string{
	computerPlayer: "C",
	personPlayer:   "P",
	-1:             " ",
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

type Move struct {
	player int
	typ    int
	row    int
	col    int
}

type Board struct {
	N             int
	M             int
	horizontal    [][]bool
	vertical      [][]bool
	owner         [][]int
	diffInScore   int
	linesPlaced   int
	lastCompletes bool
}

func buildBoard(n, m int) *Board {
	b := &Board{N: n, M: m}

	b.horizontal = make([][]bool, n+1)
	for i := range b.horizontal {
		b.horizontal[i] = make([]bool, m)
	}

	b.vertical = make([][]bool, n)
	for i := range b.vertical {
		b.vertical[i] = make([]bool, m+1)
	}

	b.owner = make([][]int, n)
	for i := range b.owner {
		b.owner[i] = make([]int, m)
		for j := range b.owner[i] {
			b.owner[i][j] = -1
		}
	}

	return b
}

func (b *Board) completed() bool {
	return b.linesPlaced == (b.N+1)*b.M+b.N*(b.M+1)
}

func (b *Board) printBoard() {
	horSym := "-"
	verSym := "|"

	fmt.Println()

	for i := 0; i <= b.N; i++ {
		for j := 0; j < b.M; j++ {
			if b.horizontal[i][j] {
				fmt.Printf("o%s", horSym)
			} else {
				fmt.Print("o ")
			}
		}
		fmt.Println("o")

		if i < b.N {
			for j := 0; j <= b.M; j++ {
				if b.vertical[i][j] {
					fmt.Printf("%s", verSym)
				} else {
					fmt.Print(" ")
				}
				if j < b.M {
					fmt.Print(playerToSymbol[b.owner[i][j]])
				}
			}
			fmt.Println()
		}
	}

	fmt.Println()
}

func (b *Board) isCompleted(row, col int) bool {
	return b.horizontal[row][col] &&
		b.vertical[row][col] &&
		b.vertical[row][col+1] &&
		b.horizontal[row+1][col]
}

func (b *Board) finalScores() (int, int) {
	comp := (b.N*b.M + b.diffInScore) / 2
	return comp, comp - b.diffInScore
}

func (b *Board) checkForUpdates(m Move) bool {
	ret := 0
	if m.typ == verticalLine {
		if m.col > 0 {
			if b.isCompleted(m.row, m.col-1) {
				b.owner[m.row][m.col-1] = m.player
				ret++
			}
		}
		if m.col < b.M {
			if b.isCompleted(m.row, m.col) {
				b.owner[m.row][m.col] = m.player
				ret++
			}
		}
	} else {
		if m.row > 0 {
			if b.isCompleted(m.row-1, m.col) {
				b.owner[m.row-1][m.col] = m.player
				ret++
			}
		}
		if m.row < b.N {
			if b.isCompleted(m.row, m.col) {
				b.owner[m.row][m.col] = m.player
				ret++
			}
		}
	}
	b.diffInScore += ret * (1 - 2*m.player)
	return ret > 0
}

func (b *Board) disregardUpdates(m Move) bool {
	ret := 0
	if m.typ == verticalLine {
		if m.col > 0 {
			if b.isCompleted(m.row, m.col-1) {
				b.owner[m.row][m.col-1] = -1
				ret++
			}
		}
		if m.col < b.M {
			if b.isCompleted(m.row, m.col) {
				b.owner[m.row][m.col] = -1
				ret++
			}
		}
	} else {
		if m.row > 0 {
			if b.isCompleted(m.row-1, m.col) {
				b.owner[m.row-1][m.col] = -1
				ret++
			}
		}
		if m.row < b.N {
			if b.isCompleted(m.row, m.col) {
				b.owner[m.row][m.col] = -1
				ret++
			}
		}
	}
	b.diffInScore -= ret * (1 - 2*m.player)
	return ret > 0
}

func (b *Board) hashState(turn int) int64 {
	hash := int64(turn)
	for i := range b.horizontal {
		for j := range b.horizontal[i] {
			hash *= 2
			if b.horizontal[i][j] {
				hash++
			}
		}
	}

	for i := range b.vertical {
		for j := range b.vertical[i] {
			hash *= 2
			if b.vertical[i][j] {
				hash++
			}
		}
	}

	for i := range b.owner {
		for j := range b.owner[i] {
			hash = hash*3 + int64(b.owner[i][j]+1)
		}
	}

	return hash

}

func (b *Board) place(m Move) error {

	b.lastCompletes = false

	if m.typ != verticalLine && m.typ != horizontalLine {
		return fmt.Errorf("no such line type")
	}

	if m.row < 0 || m.row > b.N {
		return fmt.Errorf("indices out of bounds")
	}

	if m.typ == verticalLine {
		if m.row >= b.N || m.col > b.M {
			return fmt.Errorf("indices out of bounds")
		}

		if b.vertical[m.row][m.col] {
			return fmt.Errorf("there is a stick there")
		}
		b.vertical[m.row][m.col] = true
	} else {
		if m.row > b.N || m.col >= b.M {
			return fmt.Errorf("indices out of bounds")
		}

		if b.horizontal[m.row][m.col] {
			return fmt.Errorf("there is a stick there")
		}
		b.horizontal[m.row][m.col] = true
	}

	b.lastCompletes = b.checkForUpdates(m)
	b.linesPlaced++

	return nil
}

func (b *Board) unplace(m Move) error {

	if m.typ != verticalLine && m.typ != horizontalLine {
		return fmt.Errorf("no such line type")
	}

	if m.row < 0 || m.row > b.N {
		return fmt.Errorf("indices out of bounds")
	}

	if m.typ == verticalLine {
		if m.row >= b.N || m.col > b.M {
			return fmt.Errorf("indices out of bounds")
		}

		if !b.vertical[m.row][m.col] {
			return fmt.Errorf("there isnt a stick there")
		}

		b.disregardUpdates(m)
		b.vertical[m.row][m.col] = false
	} else {
		if m.row > b.N || m.col >= b.M {
			return fmt.Errorf("indices out of bounds")
		}

		if !b.horizontal[m.row][m.col] {
			return fmt.Errorf("there isn't a stick there")
		}

		b.disregardUpdates(m)
		b.horizontal[m.row][m.col] = false
	}

	b.linesPlaced--

	return nil
}

func (b *Board) lastPlacedCompletes() bool {
	return b.lastCompletes
}

type AlphaBeta struct {
	reachedDepth bool
	maxDepth     int
	startTime    time.Time
	seen         map[[3]int64]int
}

func (a *AlphaBeta) maximize(b *Board, alpha, beta, depth int) (*Move, int) {

	if b.completed() {
		return nil, b.diffInScore
	}

	hash := [3]int64{b.hashState(computerPlayer), int64(alpha), int64(beta)}
	if cached, ok := a.seen[hash]; ok {
		return nil, cached
	}

	if depth > a.maxDepth {
		a.reachedDepth = true
		return nil, b.diffInScore
	}

	v := -math.MaxInt32
	var bestMove Move

	for t := 0; t <= 1; t++ {
		for i := 0; i <= b.N; i++ {
			for j := 0; j <= b.M; j++ {
				if time.Since(a.startTime) > minTime {
					return nil, 0
				}

				sc := 0
				m := Move{
					player: computerPlayer,
					typ:    t,
					row:    i,
					col:    j,
				}
				err := b.place(m)
				if err != nil {
					continue
				}
				if b.lastCompletes {
					_, sc = a.maximize(b, alpha, beta, depth)
				} else {
					_, sc = a.minimize(b, alpha, beta, depth+1)
				}
				b.unplace(m)

				if sc > v {
					v = sc
					bestMove = m
				}
				a.seen[hash] = v

				if v >= beta {
					return &bestMove, v
				}

				if v > alpha {
					alpha = v
				}
			}
		}
	}

	return &bestMove, v
}

func (a *AlphaBeta) minimize(b *Board, alpha, beta, depth int) (*Move, int) {

	if b.completed() {
		return nil, b.diffInScore
	}

	hash := [3]int64{b.hashState(personPlayer), int64(alpha), int64(beta)}
	if cached, ok := a.seen[hash]; ok {
		return nil, cached
	}

	if depth > a.maxDepth {
		a.reachedDepth = true
		return nil, b.diffInScore
	}

	v := math.MaxInt32
	var bestMove Move

	for t := 0; t <= 1; t++ {
		for i := 0; i <= b.N; i++ {
			for j := 0; j <= b.M; j++ {
				if time.Since(a.startTime) > minTime {
					return nil, 0
				}
				sc := 0

				m := Move{
					player: personPlayer,
					typ:    t,
					row:    i,
					col:    j,
				}
				err := b.place(m)
				if err != nil {
					continue
				}
				if b.lastCompletes {
					_, sc = a.minimize(b, alpha, beta, depth)
				} else {
					_, sc = a.maximize(b, alpha, beta, depth+1)
				}

				b.unplace(m)

				if sc < v {
					v = sc
					bestMove = m
				}

				a.seen[hash] = v

				if v <= alpha {
					return &bestMove, v
				}

				if v < beta {
					beta = v
				}
			}
		}
	}

	return &bestMove, v
}

type DotsAndDashes struct {
	N          int
	M          int
	whoseTurn  int
	turnNumber int
	board      *Board
}

func (dad *DotsAndDashes) gameOver() bool {
	return dad.board.completed()
}

func (dad *DotsAndDashes) printBoard() {
	dad.board.printBoard()
}

func (dad *DotsAndDashes) playerPlay() {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Println("Please enter your move:")

		line, err := reader.ReadString('\n')
		if err != nil {
			continue
		}
		numbers, err := retrieveNumbers(line, 3)
		if err != nil {
			fmt.Println(err)
			continue
		}

		if err := dad.board.place(Move{
			player: dad.whoseTurn,
			typ:    numbers[0] - 1,
			row:    numbers[2] - 1,
			col:    numbers[1] - 1,
		}); err != nil {
			fmt.Println(err)
			continue
		}

		break
	}

}

func (dad *DotsAndDashes) computerPlay() {
	fmt.Println("Computer plays:")

	startTime := time.Now()
	var prevMove *Move
	sc := 0
	maxDepth := 0

	for ; ; maxDepth++ {
		ab := AlphaBeta{false, maxDepth, startTime, make(map[[3]int64]int)}
		action, scc := ab.maximize(dad.board, -math.MaxInt32, math.MaxInt32, 0)
		if action != nil {
			prevMove = action
			sc = scc
		}
		if action == nil || !ab.reachedDepth {
			break
		}
	}
	fmt.Println("REACHED DEPTH: ", maxDepth)
	fmt.Println("ESTIMATED SCORE: ", sc)
	dad.board.place(*prevMove)
}

func (dad *DotsAndDashes) Play() {
	reader := bufio.NewReader(os.Stdin)

	// find N & M
	for {
		fmt.Println("Enter N, M:")

		line, err := reader.ReadString('\n')
		if err != nil {
			continue
		}
		numbers, err := retrieveNumbers(line, 2)
		if err != nil {
			fmt.Println(err)
			continue
		}
		dad.N, dad.M = numbers[0], numbers[1]
		break
	}

	dad.board = buildBoard(dad.N, dad.M)

	// find who plays first
	for {
		fmt.Println("Computer(1) or Player(2) is first?")

		line, err := reader.ReadString('\n')
		if err != nil {
			continue
		}
		numbers, err := retrieveNumbers(line, 1)
		if err != nil {
			fmt.Println(err)
			continue
		}
		if numbers[0] != 1 && numbers[0] != 2 {
			continue
		}

		dad.whoseTurn = numbers[0] - 1
		break
	}

	for !dad.gameOver() {
		dad.printBoard()
		if dad.whoseTurn == computerPlayer {
			dad.computerPlay()
		} else {
			dad.playerPlay()
		}
		if !dad.board.lastPlacedCompletes() {
			dad.whoseTurn = 1 - dad.whoseTurn
		} else if dad.whoseTurn == computerPlayer {
			time.Sleep(time.Second)
		}
		dad.turnNumber++
	}

	dad.printBoard()

	if dad.board.diffInScore > 0 {
		fmt.Println("WINNER IS COMPUTER")
	} else if dad.board.diffInScore < 0 {
		fmt.Println("WINNER IS PLAYER")
	} else {
		fmt.Println("NO ONE IS WINNER")
	}

	comp, pl := dad.board.finalScores()
	fmt.Println("--SCORES--")
	fmt.Printf("COMPUTER: [%v]\n", comp)
	fmt.Printf("PLAYER: [%v]\n", pl)
}

func main() {
	dad := DotsAndDashes{}
	dad.Play()
}
