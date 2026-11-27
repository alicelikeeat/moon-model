#!/usr/bin/env bash
# Downloads the CC0 MakeHuman (hm08) assets used to rebuild the Seraph body.
# MakeHuman assets (base mesh, targets, skeleton, weights, eye proxy) are CC0 1.0:
# https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.md (section C).
set -euo pipefail
DEST="${1:-$(dirname "$0")/mh}"
REV="${MH_REV:-master}"
R="https://raw.githubusercontent.com/makehumancommunity/makehuman/$REV/makehuman/data"
mkdir -p "$DEST/eyes"
get() { curl -fsSL -o "$DEST/$2" "$R/$1"; echo "  $2"; }
get 3dobjs/base.obj base.obj
get rigs/default.mhskel default.mhskel
get rigs/default_weights.mhw default_weights.mhw
get targets/macrodetails/proportions/female-young-averagemuscle-averageweight-idealproportions.target proportions.target
get targets/macrodetails/universal-female-young-averagemuscle-minweight.target slim.target
get targets/breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness.target bust.target
get eyes/high-poly/high-poly.obj eyes/high-poly.obj
get eyes/high-poly/high-poly.mhclo eyes/high-poly.mhclo
mkdir -p "$DEST/face"
for t in head/head-oval head/head-scale-vert-decr chin/chin-height-decr chin/chin-width-decr \
         nose/nose-scale-horiz-decr nose/nose-scale-vert-decr nose/nose-point-width-decr \
         mouth/mouth-upperlip-volume-incr mouth/mouth-lowerlip-volume-incr mouth/mouth-scale-horiz-decr \
         mouth/mouth-cupidsbow-incr eyes/l-eye-scale-incr eyes/r-eye-scale-incr eyes/l-eye-corner1-up \
         eyes/r-eye-corner1-up cheek/l-cheek-bones-incr cheek/r-cheek-bones-incr \
         forehead/forehead-scale-vert-decr neck/neck-scale-horiz-decr; do
  get "targets/$t.target" "face/$(basename "$t").target"
done
