"""Initial schema migration for AI Bettor."""

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    # 1. matches table
    op.create_table(
        "matches",
        sa.Column("match_id", sa.String(50), primary_key=True, comment="Unique match ID from API"),
        sa.Column("home_team", sa.String(100), nullable=False, comment="Home team name"),
        sa.Column("away_team", sa.String(100), nullable=False, comment="Away team name"),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False, comment="Match kickoff time"),
        sa.Column("league", sa.String(100), nullable=False, comment="League name"),
        sa.Column("sport", sa.String(50), nullable=False, server_default="football", comment="Sport type"),
        sa.Column("status", sa.String(20), nullable=False, default="upcoming", comment="Match status"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), comment="Created timestamp"),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.text("NOW()"), comment="Updated timestamp"),
    )

    # 2. bookmakers table
    op.create_table(
        "bookmakers",
        sa.Column("bookmaker_id", sa.String(50), primary_key=True, comment="Bookmaker ID"),
        sa.Column("name", sa.String(100), nullable=False, comment="Bookmaker name"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true"), comment="Whether bookmaker is active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # 3. markets table
    op.create_table(
        "markets",
        sa.Column("market_id", sa.String(50), primary_key=True, comment="Market ID"),
        sa.Column("name", sa.String(100), nullable=False, comment="Market name"),
        sa.Column("key", sa.String(50), nullable=False, unique=True, comment="Market key"),
        sa.Column("sport", sa.String(50), nullable=False, server_default="football", comment="Sport"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )

    # 4. odds_snapshots table
    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True, comment="Snapshot ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=False, comment="Match ID"),
        sa.Column("bookmaker", sa.String(50), nullable=False, comment="Bookmaker name"),
        sa.Column("market", sa.String(50), nullable=False, comment="Market key"),
        sa.Column("selection", sa.String(100), nullable=False, comment="Selection/outcome"),
        sa.Column("line", sa.String(20), nullable=True, comment="Handicap or OU line"),
        sa.Column("odds", sa.Float(), nullable=False, comment="Decimal odds"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), comment="Snapshot timestamp"),
    )

    # 5. predictions table
    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.String(50), primary_key=True, comment="Prediction ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("decision", sa.String(10), nullable=False, comment="BET or NO BET"),
        sa.Column("market", sa.String(50), nullable=True, comment="Market type"),
        sa.Column("selection", sa.String(100), nullable=True, comment="Selected outcome"),
        sa.Column("odds", sa.Float(), nullable=True, comment="Odds"),
        sa.Column("bookmaker", sa.String(50), nullable=True, comment="Bookmaker"),
        sa.Column("model_probability", sa.Float(), nullable=False, comment="Model probability"),
        sa.Column("implied_probability", sa.Float(), nullable=False, comment="Market implied probability"),
        sa.Column("edge", sa.Float(), nullable=False, comment="Edge"),
        sa.Column("ev", sa.Float(), nullable=False, comment="Expected value"),
        sa.Column("confidence_score", sa.Integer(), nullable=False, comment="Confidence 0-100"),
        sa.Column("risk_level", sa.String(20), nullable=False, comment="Risk level"),
        sa.Column("reasoning", sa.Text(), nullable=True, comment="Reasoning"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # 6. simulations table
    op.create_table(
        "simulations",
        sa.Column("simulation_id", sa.String(50), primary_key=True, comment="Simulation ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("home_win_probability", sa.Float(), nullable=False, comment="Home win probability"),
        sa.Column("draw_probability", sa.Float(), nullable=False, comment="Draw probability"),
        sa.Column("away_win_probability", sa.Float(), nullable=False, comment="Away win probability"),
        sa.Column("handicap_probability", sa.Float(), nullable=True, comment="Handicap probability"),
        sa.Column("over_probability", sa.Float(), nullable=True, comment="Over probability"),
        sa.Column("under_probability", sa.Float(), nullable=True, comment="Under probability"),
        sa.Column("variance", sa.Float(), nullable=False, comment="Variance"),
        sa.Column("stability", sa.Float(), nullable=False, comment="Stability score"),
        sa.Column("simulation_count", sa.Integer(), nullable=False, server_default=sa.text("20000"), comment="Number of simulations"),
        sa.Column("random_seed", sa.Integer(), nullable=True, comment="Random seed used"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # 7. agent_analyses table
    op.create_table(
        "agent_analyses",
        sa.Column("analysis_id", sa.String(50), primary_key=True, comment="Analysis ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("agent_type", sa.String(50), nullable=False, comment="Agent type (data_scout, quant_analyst, market_analyst, simulation_analyst, risk_manager, better_brain)"),
        sa.Column("status", sa.String(50), nullable=False, comment="Agent status"),
        sa.Column("output", sa.JSON(), nullable=True, comment="Agent output JSON"),
        sa.Column("execution_time", sa.Float(), nullable=True, comment="Execution time in seconds"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="Error message if any"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # 8. risk_assessments table
    op.create_table(
        "risk_assessments",
        sa.Column("assessment_id", sa.String(50), primary_key=True, comment="Risk assessment ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=True),
        sa.Column("bankroll_risk_percent", sa.Float(), nullable=False, comment="Bankroll risk percentage"),
        sa.Column("exposure", sa.Float(), nullable=False, comment="Current exposure"),
        sa.Column("drawdown", sa.Float(), nullable=False, comment="Current drawdown"),
        sa.Column("correlation_risk", sa.Float(), nullable=False, comment="Correlation risk"),
        sa.Column("risk_level", sa.String(20), nullable=False, comment="Risk level"),
        sa.Column("veto_decision", sa.Boolean(), nullable=False, server_default=sa.text("false"), comment="Whether risk manager vetoed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    # 9. bets table
    op.create_table(
        "bets",
        sa.Column("bet_id", sa.String(50), primary_key=True, comment="Bet ID"),
        sa.Column("match_id", sa.String(50), sa.ForeignKey("matches.match_id"), nullable=True),
        sa.Column("decision", sa.String(10), nullable=False, comment="BET or NO BET"),
        sa.Column("market", sa.String(50), nullable=True, comment="Market type"),
        sa.Column("selection", sa.String(100), nullable=True, comment="Selection"),
        sa.Column("odds", sa.Float(), nullable=True, comment="Odds"),
        sa.Column("bookmaker", sa.String(50), nullable=True, comment="Bookmaker"),
        sa.Column("stake", sa.Float(), nullable=False, comment="Stake amount"),
        sa.Column("potential_profit", sa.Float(), nullable=False, comment="Potential profit"),
        sa.Column("status", sa.String(20), nullable=False, default="pending", comment="Bet status"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True, comment="Settlement time"),
        sa.Column("result", sa.String(20), nullable=True, comment="Win/Loss/Push"),
    )

    # 10. bankroll table
    op.create_table(
        "bankroll",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True, comment="Bankroll ID"),
        sa.Column("current_balance", sa.Float(), nullable=False, server_default=sa.text("1000.0"), comment="Current bankroll"),
        sa.Column("total_staked", sa.Float(), nullable=False, server_default=sa.text("0.0"), comment="Total staked"),
        sa.Column("total_won", sa.Float(), nullable=False, server_default=sa.text("0.0"), comment="Total won"),
        sa.Column("total_profit", sa.Float(), nullable=False, server_default=sa.text("0.0"), comment="Total profit/loss"),
        sa.Column("roi", sa.Float(), nullable=False, server_default=sa.text("0.0"), comment="ROI percentage"),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.text("NOW()")),
    )

    # 11. system_logs table
    op.create_table(
        "system_logs",
        sa.Column("log_id", sa.BigInteger(), sa.Identity(), primary_key=True, comment="Log ID"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), comment="Log timestamp"),
        sa.Column("service", sa.String(50), nullable=False, comment="Service name"),
        sa.Column("agent", sa.String(50), nullable=True, comment="Agent name"),
        sa.Column("match_id", sa.String(50), nullable=True, comment="Match ID"),
        sa.Column("action", sa.String(100), nullable=False, comment="Action performed"),
        sa.Column("status", sa.String(50), nullable=False, comment="Status"),
        sa.Column("latency", sa.Float(), nullable=True, comment="Execution latency"),
        sa.Column("error_details", sa.Text(), nullable=True, comment="Error details"),
    )


def downgrade() -> None:
    op.drop_table("system_logs")
    op.drop_table("bankroll")
    op.drop_table("bets")
    op.drop_table("risk_assessments")
    op.drop_table("agent_analyses")
    op.drop_table("simulations")
    op.drop_table("predictions")
    op.drop_table("odds_snapshots")
    op.drop_table("markets")
    op.drop_table("bookmakers")
    op.drop_table("matches")