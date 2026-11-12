Draw Distance Grok v1
Breath of the Wild (Switch) 1.9.0 — placed-object pop-in

Install
  Copy this folder to:
    <Lichii user directory>/load/01007EF00011E000/Draw Distance Grok v1/
  so the emulator sees:
    load/01007EF00011E000/Draw Distance Grok v1/exefs/
  Enable the mod in the game's add-ons. It applies only when the main NSO
  build id matches the @nsobid line.

This folder does not contain a working byte patch. See below.

What the pop-in is
  Enemies, camps, cauldrons, chests and other placed actors appear about
  6–9 m in front of Link. Terrain, tree billboards, grass and texture LOD
  are different systems and are not this.

  The retail game hardcodes the placed-object draw-distance multiplier to
  1.0. In the public decompilation that value is what
  ksys::map::Object::getDispDistance returns; it is a debug multiplier the
  shipping build never raises. Cemu's Draw Distance pack and the public
  1.5.0 / 1.6.0 notes change that 1.0 (and, separately, an actor multiplier
  stored as the float 0.7 at a field written by setActorDrawDistanceMultiplier).
  Raising only the placed-object 1.0 is what pulls camps and props out.
  1.5x is the strength those patches treat as the safe step. 2.0x keeps
  more actors loaded and can stutter or run out of memory.

  On 1.6.0 (build id 8E9978D50BDD20B4C8395A106C27FFDE) public notes put
  setActorDrawDistanceMultiplier at 0x7101586b5c and the object multiplier
  (v33 = 1.0) at 0x7101557700. 1.9.0 was rebuilt (CSPGO); those addresses
  do not apply. No public file offset for the 1.9.0 instruction was found,
  so this folder does not guess one. A wrong word in main corrupts the game.

Build id
  Community notes for the 1.9.0 main executable (Nexus "BotW 1.9.0 - Draw
  Distance", and Savage Wind's 1.9.0 setup check) give:

    CD57B23FA4BBAD65803D9788C01821EE

  ZeldaMods' version table does not list 1.9.0's build id. Confirm it in
  Lichii's log. While the game boots, the loader prints one line per NSO:

    Patching NSO for name=<module>, build_id=<hex>

  The game code is name=main (or main.nso). The hex is the build id, with
  trailing zeroes stripped. It must match @nsobid in the pchtxt or Lichii
  ignores the file. If it differs, replace the @nsobid line; do not add an
  offset until the 1.0 immediate in getDispDistance is known for that id.

Cost
  A 1.5x object multiplier keeps more actors resident. Expect more CPU time
  on the actor update and more memory, not a change to the GPU thread's
  per-draw cost. It does not change terrain mesh LOD.

Not an emulator bug
  The same pop-in is in official build 160 with every speedup switch off.
  See the round 4 report for which switches could theoretically hide an
  object for another reason (only fence-download off, and only for a buffer
  the CPU reads). That is not this distance.
