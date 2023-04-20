DIE()
{
  echo ERROR:$1
  exit 1
}


cargo check || DIE
cargo clippy || DIE
cargo test || DIE
cargo bench --no-run --profile dev || DIE
