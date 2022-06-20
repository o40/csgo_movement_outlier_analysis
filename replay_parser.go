// Parser that outputs player position and viewangles as csv data.
// Example usage: ./replay_parser.exe -demo=path/to/replay.dem > output.csv

package main

import (
	"fmt"
	"os"
	"flag"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// Print position and viewangles for each player in the replay as csv.
func outputPlayerPositionAndViewAnglesAsCsv(parser dem.Parser) {
	// Filter out ticks where the player is scoped in since it affects
	// the minimum step size (in pitch and yaw) generated by the mouse.
	lastScopedTick := make(map[string]int)

	parser.RegisterEventHandler(func(e events.FrameDone) {
		players := parser.GameState().Participants().Playing()
		for _, player := range players {

			// Save the last frame where the player was scoped in
			if player != nil && player.IsScoped() {
				lastScopedTick[player.Name] = parser.GameState().IngameTick()
			}

			value, ok := lastScopedTick[player.Name]

			// Only save data where the player exists, is alive and has not been scoped for a few frames
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

	outputPlayerPositionAndViewAnglesAsCsv(p)
}