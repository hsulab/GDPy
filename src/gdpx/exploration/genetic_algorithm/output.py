"""Scrolling GA progress blocks, independent of structure ownership and RNGs."""
import time

from gdpx.core.output import Box


class GenerationReporter(Box):
    def __init__(self, database, generation, maximum_generation, size, objective,
                 resumed=False, emit=None, unicode=None):
        self.database = database
        self.generation = generation
        self.size = size
        self.objective = objective
        self.started = time.monotonic()
        self.closed = False
        title = f'genetic algorithm | generation {generation}/{maximum_generation} | '
        title += 'initialization' if generation == 0 else 'evolution'
        if resumed:
            title += ' | resumed'
        super().__init__(title, emit=emit, unicode=unicode)
        self.line(f'candidates: {size}   objective: {objective} [eV]')
        if resumed:
            info = database.get_generation_info(generation)
            self.line(f'resumed with {info.num_relaxed}/{size} evaluations committed')

    def finish(self, status, detail=None):
        if self.closed:
            return
        # Read committed scores without loading structures or touching population state.
        candidates = {}
        for row in self.database.connection.select(
                relaxed=1, sort='id', columns=['id', 'key_value_pairs']):
            if row.get('generation', 0) <= self.generation:
                candidates[row.get('confid', row.id)] = row
        rows = list(candidates.values())
        current = [row for row in rows if row.get('generation', 0) == self.generation]
        eligible = [row for row in rows if not row.get('extinct', 0)]
        scores = [row.get('target') for row in eligible if row.get('target') is not None]
        best = f'{min(scores):.4f} eV' if scores else '—'
        self.border('middle')
        self.line(f'{status.lower()} | evaluated: {len(current)}/{self.size} | '
                  f'extinct: {sum(bool(row.get("extinct", 0)) for row in current)}')
        self.line(f'eligible candidates so far: {len(eligible)}')
        self.line(f'best eligible {self.objective}: {best}')
        if detail:
            self.line(detail)
        self.line(f'elapsed this invocation: {time.monotonic() - self.started:.1f} s')
        self.border('bottom')
        self.closed = True


def report_setup(population_config, operators, objective):
    box = Box('genetic algorithm | setup')
    box.line(f'initial candidates: {population_config.init_size} | '
             f'candidates/generation: {population_config.gen_size}')
    box.line(f'retained population: {population_config.retained_size} | objective: {objective} [eV]')
    for group, settings in operators.items():
        pairing = settings.get('pairing')
        if pairing is not None:
            box.line(f'{group} crossover: {type(pairing).__name__}')
        mutations = settings.get('mutations')
        if mutations is not None:
            for mutation in mutations.oplist:
                box.line(f'{group} mutation: {type(mutation).__name__}')
    box.border('bottom')
