import os
import pickle
from datetime import datetime

import torch
import numpy as np


def loadfile(filename, verbose=False):
    if verbose:
        print(f"Loading {filename}")
    return pickle.load(open(filename, "rb"))

def savefile(filename, data, metadata={}, verbose=False):
    if verbose:
        print(f"Saving {filename}")
    metadata.update({"saved_on": datetime.now().strftime("%d-%m-%Y_%H-%M-%S")})
    pickle.dump((data, metadata), open(filename, "wb+"))

def save_ovito(filename, traj, species=None, lattice=None, length=None):
    print(f"Saving ovito file: {filename}")
    with open(filename, "w+") as ofile:
        for state in traj:
            N, dim = state['position'].shape
            if species is None:
                species = torch.ones(N, dtype=torch.int).reshape(-1, 1)
            else:
                species = torch.tensor(species).reshape(-1, 1)

            hinting = f"Properties=id:I:1:species:R:1:pos:R:{dim}:vel:R:{dim}:force:R:{dim}"
            tmp = torch.eye(dim).flatten()
            if length is not None:
                lattice = " ".join(
                    [(f"{length}" if i != 0 else "0") for i in tmp])
                Lattice = f'Lattice="{lattice}"'
            if lattice is not None:
                Lattice = f'Lattice="{lattice}"'
            str_ = f"{N}" + f"\n{Lattice} {hinting}\n"
            ofile.write(str_)
            data = torch.cat(
                [species, state['position'], state['velocity'], state['force']], dim=1)
            for j in range(N):
                line = "\t".join([str(item) for item in data[j, :]])
                str_ = f"{j+1}\t" + line + "\n"
                ofile.write(str_)

def write_ovito(filename, g, box, atoms=None, species=None, comment="No comment given."):
    with open(filename, "w+") as f:
        L = box["x"]["x"]

        if species is None:
            species = torch.cat([(ind+1)*torch.ones(v, dtype=torch.int)
                                  for ind, v in enumerate(atoms.values())])
            species.flatten_()
        
        types = species

        atms = len(atoms)
        N = len(species)
        f.write(f"{comment}\n\n")
        f.write(f"{g['n_node'].sum()} atoms\n")
        f.write(f"{atms} atom types\n")
        f.write(f"{len(g['senders'])} bonds\n")
        f.write(f"{atms*(atms+1)//2} bond types\n\n")
        f.write(f"{0} {L} xlo xhi\n")
        f.write(f"{0} {L} ylo yhi\n")
        f.write(f"{0} {L} zlo zhi\n\n")
        f.write("Atoms #full\n\n")

        def btype(i, j, s):
            return s[i-1]+j-i+1 if i <= j else s[j-1]+i-j+1

        def get_s(atms):
            s = [0]
            for i in range(1, atms):
                s += [s[-1] + atms-i+1]
            return s

        for i, t, row, vel in zip(range(N), types, g["nodes"]["position"],  g["nodes"]["velocity"]):
            f.write(f"{i+1} 1 {t} 0.0 "+" ".join(map(str, row)) +
                    " "+" ".join(map(str, vel))+"\n")

        _s = get_s(atms)
        f.write("\nBonds #full\n\n")
        senders = g["senders"]
        receivers = g["receivers"]
        for i, s, r in zip(range(len(senders)), senders, receivers):
            f.write(f"{i+1} {btype(types[s], types[r], _s)} {s+1} {r+1}\n")

class write_dump():
    def checkfile(self, filename):
        if os.path.isfile(filename):
            dir_ = os.path.dirname(filename)
            base = os.path.basename(filename)
            tag = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            return self.checkfile(dir_+f"/{tag}_"+base)
        else:
            return filename

    def __init__(self, filename, timestep=0):
        self.filename = self.checkfile(filename)
        self.timestep = timestep
        print(f"Writing to {self.filename}")

    def __call__(self, g, box, atoms=None, species=None, timestep=None, comment="No comment given."):
        filename = self.filename
        if timestep is not None:
            self.timestep = timestep
        with open(filename, "a") as f:
            L = box["x"]["x"]

            if species is None:
                species = torch.cat([(ind+1)*torch.ones(v, dtype=torch.int)
                                      for ind, v in enumerate(atoms.values())])
                species.flatten_()
            
            types = species

            atms = len(atoms)
            N = len(species)

            f.write(f"ITEM: TIMESTEP\n{self.timestep}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{N}\n")

            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write(f"{0} {L} 0.0\n")
            f.write(f"{0} {L} 0.0\n")
            f.write(f"{0} {L} 0.0\n")

            def btype(i, j, s):
                return s[i-1]+j-i+1 if i <= j else s[j-1]+i-j+1

            def get_s(atms):
                s = [0]
                for i in range(1, atms):
                    s += [s[-1] + atms-i+1]
                return s

            f.write("ITEM: ATOMS id type x y z vx vy vz\n")
            for i, t, row, vel in zip(range(N), types, g["nodes"]["position"],  g["nodes"]["velocity"]):
                f.write(f"{i+1} {t} "+" ".join(map(str, row)) +
                        " "+" ".join(map(str, vel))+"\n")

            self.timestep += 1

def read_dump(filename, tags=["Atoms"]):
    def check(line):
        for tag in tags:
            if tag in line:
                return True, tag
        return False, None
    
    with open(filename, "r") as f:
        collection = {tag: [] for tag in tags}
        collect = False
        for line in f:
            if line == "\n":
                continue
            if collect:
                if line[0].isalpha():
                    iftag, tag = check(line)
                    if iftag:
                        collection[tag] += [[]]
                    else:
                        collect = False
                else:
                    collection[tag][-1] += [line]
            else:
                iftag, tag = check(line)
                if iftag:
                    collect = True
                    collection[tag] += [[]]
    return {k: torch.tensor([[float(j) for j in _.split()] for _ in io for i in v]) for k, v in collection.items()}
