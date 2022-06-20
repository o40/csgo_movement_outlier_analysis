package main

import (
	"fmt"
	"os"
	"flag"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
	// common "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
)

func adjustVerticalAngle(angle float32) float32 {
	if (angle > 180) {
		return (angle - 360)
	}
	return angle
}

func outputPlayerPositionAndViewDirectionAsCsv(parser dem.Parser) {
	// Only do this for players that are alive and not zoomed in
	lastScopedTick := make(map[string]int)

	parser.RegisterEventHandler(func(e events.FrameDone) {
		players := parser.GameState().Participants().Playing()
		for _, player := range players {

			// It takes a few ticks to zoom out. During this time the sensitivity is not fully zoomed out either.
			if player != nil && player.IsScoped() {
				lastScopedTick[player.Name] = parser.GameState().IngameTick()
			}

			value, ok := lastScopedTick[player.Name]

			if player != nil && player.IsAlive() && !player.IsScoped() && (!ok || value < parser.GameState().IngameTick() - 5) {

				fmt.Printf("%s,%d,%.13f,%.13f,%.13f,%.13f\n",
						player.Name,
						parser.GameState().IngameTick(),
						player.ViewDirectionX(),
						player.ViewDirectionY(),
						player.Position().X,
						player.Position().Y)
			}
		}
	})

	fmt.Printf("%s,%s,%s,%s,%s,%s\n", "name", "tick", "yaw", "pitch", "x", "y")
	parser.ParseToEnd()

}

func main() {
	demoPathPtr := flag.String("demo", "", "Path to the demo")
	flag.Parse()

	// Mandatory argument
	if *demoPathPtr == "" {
		fmt.Printf("Missing argument demo\n")
		os.Exit(1)
	}

	if _, err := os.Stat(*demoPathPtr); os.IsNotExist(err) {
  		fmt.Printf("%s does not exist\n", *demoPathPtr)
  		os.Exit(1)
	}

	f, err := os.Open(*demoPathPtr)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := dem.NewParser(f)

	outputPlayerPositionAndViewDirectionAsCsv(p)
}