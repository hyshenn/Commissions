# Commissions Macro

## How It Works

- Step 1: Fetches current commissions from the tab list, only choosing viable ones.
- Step 2: Travels to the commission using Theta* pathfinding.
- Step 3: Starts the mining thread loop.
- Step 4: Fetches the finished commission, warps to the Forge, and claims the commission.
 
## Mining System Functionality

Raycasting - The mining system uses raycasting to find visible blocks and their interception points, calculating their yaw and pitch to ensure we don't overshoot or miss.
Threaded Scanning - The reason we scan blocks on a different thread is due to MineScript's slow computational speed. To combat this, we implemented two lists: one for the current blocks and another for when we run out of blocks.

## Warps & Travel Positions

Travel Waypoints - We decided to use hand-picked waypoints for the best compatibility with our scripts, ensuring optimal detection and success rates.
Warps - For warps, we also use a raycasting system to determine the correct position to aim at.

## Kill Commissions

While this isn't perfect, it works. This will change in the future — currently, we're using a static pathfinder, but this is expected to improve over time.

## Credits
Pathfinder & rotation: https://github.com/Jones0073/AShortPath
