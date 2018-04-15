#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;
use mpi::datatype::{MutView, UserDatatype, View};
use mpi::Count;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);

    let count = world.size() as usize;
    let i = 2_u64.pow(world.rank() as u32 + 1);

    if world.rank() == root_rank {
        let mut a = vec![0u64; count];
        mpi::request::scope(|_scope| {
            root_process
                .immediate_gather_into_root(&i, &mut a[..])
                .wait();
        });
        println!("Root gathered sequence: {:?}.", a);
        assert!(
            a.iter()
                .enumerate()
                .all(|(a, &b)| b == 2u64.pow(a as u32 + 1))
        );
    } else {
        mpi::request::scope(|_scope| {
            root_process.immediate_gather_into(&i).wait();
        });
    }

    let factor = world.rank() as u64 + 1;
    let a = (1_u64..)
        .take(count)
        .map(|x| x * factor)
        .collect::<Vec<_>>();

    if world.rank() == root_rank {
        let mut t = vec![0u64; count * count];
        mpi::request::scope(|_scope| {
            root_process
                .immediate_gather_into_root(&a[..], &mut t[..])
                .wait();
        });
        println!("Root gathered table:");
        for r in t.chunks(count) {
            println!("{:?}", r);
        }
        assert!(
            (0_u64..)
                .zip(t.iter())
                .all(|(a, &b)| b == (a / count as u64 + 1) * (a % count as u64 + 1))
        );
    } else {
        mpi::request::scope(|_scope| {
            root_process.immediate_gather_into(&a[..]).wait();
        });
    }

    let d = UserDatatype::contiguous(count as Count, &u64::equivalent_datatype());
    let sv = unsafe { View::with_count_and_datatype(&a[..], 1, &d) };

    if world.rank() == root_rank {
        let mut t = vec![0u64; count * count];

        {
            let mut rv =
                unsafe { MutView::with_count_and_datatype(&mut t[..], count as Count, &d) };
            mpi::request::scope(|_scope| {
                root_process.immediate_gather_into_root(&sv, &mut rv).wait();
            });
        }

        println!("Root gathered table:");
        for r in t.chunks(count) {
            println!("{:?}", r);
        }
        assert!(
            (0_u64..)
                .zip(t.iter())
                .all(|(a, &b)| b == (a / count as u64 + 1) * (a % count as u64 + 1))
        );
    } else {
        mpi::request::scope(|_scope| {
            root_process.immediate_gather_into(&sv).wait();
        });
    }
}
